#include "llvm_lowering.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace forge::detail {
namespace {

llvm::Value* emit_value(llvm::IRBuilder<>& builder, Op op,
                        llvm::ArrayRef<llvm::Value*> operands) {
    switch (op) {
    case Op::add: return builder.CreateFAdd(operands[0], operands[1], "sum");
    case Op::multiply: return builder.CreateFMul(operands[0], operands[1], "product");
    case Op::sqrt:
        return builder.CreateIntrinsic(llvm::Intrinsic::sqrt, {builder.getFloatTy()}, operands);
    default: throw std::logic_error("Expected a pointwise arithmetic operation");
    }
}

llvm::Function* emit_loop(llvm::Module& module, const Graph& graph, const LoopRegion& region,
                          llvm::ArrayRef<llvm::Value*> arguments, bool simd) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* index_type = builder.getInt64Ty();
    auto* float_type = builder.getFloatTy();
    std::vector<llvm::Type*> parameters{builder.getPtrTy(), index_type};
    for (auto* argument : arguments) {
        parameters.push_back(argument->getType());
    }

    auto* type = llvm::FunctionType::get(builder.getVoidTy(), parameters, false);
    const auto name = region.operations.empty() ? "copy" :
        region.operations.size() == 1 ? op_name(graph.nodes()[region.output].op) : "fused";
    auto* function = llvm::Function::Create(type, llvm::Function::InternalLinkage,
        std::string("forge_") + name + '_' + std::to_string(region.output), module);
    // Preserve the planned loop boundaries in subsequent LLVM passes.
    function->addFnAttr(llvm::Attribute::NoInline);
    function->addFnAttr(llvm::Attribute::NoBuiltin);
    // The launcher proves output/read-buffer disjointness. Read inputs may overlap.
    auto* output = function->getArg(0);
    output->setName("output");
    output->addAttr(llvm::Attribute::NoAlias);
    auto* count = function->getArg(1);
    count->setName("count");

    auto* entry = llvm::BasicBlock::Create(context, "entry", function);
    auto* loop = llvm::BasicBlock::Create(context, "loop", function);
    auto* exit = llvm::BasicBlock::Create(context, "exit", function);
    builder.SetInsertPoint(entry);
    builder.CreateCondBr(builder.CreateICmpEQ(count, builder.getInt64(0)), exit, loop);

    builder.SetInsertPoint(loop);
    auto* index = builder.CreatePHI(index_type, 2, "index");
    index->addIncoming(builder.getInt64(0), entry);
    std::vector<llvm::Value*> values(graph.nodes().size());
    for (unsigned arg = 0; arg < region.arguments.size(); ++arg) {
        auto* value = function->getArg(arg + 2);
        value->setName("operand" + std::to_string(arg));
        if (value->getType()->isPointerTy()) {
            value->addAttr(llvm::Attribute::ReadOnly);
            auto* address = builder.CreateInBoundsGEP(float_type, value, index);
            values[region.arguments[arg]] = builder.CreateAlignedLoad(float_type, address, llvm::Align(1));
        } else {
            values[region.arguments[arg]] = value;
        }
    }
    for (const auto id : region.operations) {
        const auto& node = graph.nodes()[id];
        llvm::SmallVector<llvm::Value*, 2> operands;
        for (const auto operand : node.operands) {
            operands.push_back(values[operand]);
        }
        values[id] = emit_value(builder, node.op, operands);
    }
    auto* address = builder.CreateInBoundsGEP(float_type, output, index);
    builder.CreateAlignedStore(values[region.output], address, llvm::Align(1));
    auto* next = builder.CreateNUWAdd(index, builder.getInt64(1), "next");
    index->addIncoming(next, loop);
    auto* backedge = builder.CreateCondBr(builder.CreateICmpEQ(next, count), exit, loop);
    if (simd) {
        const auto hint = [&](const char* name, unsigned value) {
            return llvm::MDNode::get(context, {llvm::MDString::get(context, name),
                llvm::ConstantAsMetadata::get(builder.getInt32(value))});
        };
        // Request eight floats per vector and one vector per iteration.
        auto* metadata = llvm::MDNode::getDistinct(context, {nullptr,
            hint("llvm.loop.vectorize.width", 8), hint("llvm.loop.interleave.count", 1)});
        metadata->replaceOperandWith(0, metadata);
        backedge->setMetadata(llvm::LLVMContext::MD_loop, metadata);
    }
    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
    return function;
}

} // namespace

void lower_kernel(llvm::Module& module, const Graph& graph, const KernelPlan& plan, bool simd) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* ptr_type = builder.getPtrTy();
    auto* type = llvm::FunctionType::get(builder.getVoidTy(),
        {builder.getInt64Ty(), ptr_type, ptr_type}, false);
    auto* function = llvm::Function::Create(type, llvm::Function::ExternalLinkage,
                                           "forge_kernel", module);
    auto* count = function->getArg(0);
    auto* inputs = function->getArg(1);
    auto* buffers = function->getArg(2);
    count->setName("count");
    inputs->setName("inputs");
    buffers->setName("buffers");
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));

    const auto pointer_at = [&](llvm::Value* table, std::size_t index) {
        return builder.CreateLoad(ptr_type,
            builder.CreateInBoundsGEP(ptr_type, table, builder.getInt64(index)));
    };
    const auto& nodes = graph.nodes();
    std::vector<llvm::Value*> values(nodes.size());
    auto next_region = plan.regions.begin();
    for (NodeId id = 0; id < nodes.size(); ++id) {
        if (!plan.used[id]) {
            continue;
        }
        const auto& node = nodes[id];
        if (node.op == Op::input) {
            values[id] = pointer_at(inputs, node.input_index);
        } else if (node.op == Op::literal) {
            values[id] = llvm::ConstantFP::get(builder.getFloatTy(), node.literal_value);
        } else if (node.op == Op::splat) {
            // The loop receives this value directly as a scalar argument.
            values[id] = values[node.operands[0]];
        } else if (node.shape.empty()) {
            llvm::SmallVector<llvm::Value*, 2> operands;
            for (const auto operand : node.operands) {
                operands.push_back(values[operand]);
            }
            values[id] = emit_value(builder, node.op, operands);
        }
        if (next_region != plan.regions.end() && next_region->output == id) {
            llvm::SmallVector<llvm::Value*, 4> arguments;
            for (const auto operand : next_region->arguments) {
                arguments.push_back(values[operand]);
            }
            auto* loop = emit_loop(module, graph, *next_region, arguments, simd);
            auto* output = pointer_at(buffers, plan.slots[id]);
            llvm::SmallVector<llvm::Value*, 6> call_arguments{output, count};
            call_arguments.append(arguments.begin(), arguments.end());
            builder.CreateCall(loop, call_arguments);
            values[id] = output;
            ++next_region;
        }
    }
    builder.CreateRetVoid();
}

} // namespace forge::detail
