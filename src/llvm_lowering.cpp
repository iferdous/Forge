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

void set_loop_options(llvm::CondBrInst* backedge, bool vectorize) {
    auto& context = backedge->getContext();
    const auto option = [&](const char* name, llvm::ConstantInt* value) {
        return llvm::MDNode::get(context, {llvm::MDString::get(context, name),
                                         llvm::ConstantAsMetadata::get(value)});
    };
    llvm::SmallVector<llvm::Metadata*, 3> options{nullptr};
    if (vectorize) {
        auto* i32 = llvm::Type::getInt32Ty(context);
        // Request eight floats per vector and one vector per iteration.
        options.push_back(option("llvm.loop.vectorize.width", llvm::ConstantInt::get(i32, 8)));
        options.push_back(option("llvm.loop.interleave.count", llvm::ConstantInt::get(i32, 1)));
    } else {
        options.push_back(option("llvm.loop.vectorize.enable", llvm::ConstantInt::getFalse(context)));
    }
    auto* metadata = llvm::MDNode::getDistinct(context, options);
    metadata->replaceOperandWith(0, metadata);
    backedge->setMetadata(llvm::LLVMContext::MD_loop, metadata);
}

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
        set_loop_options(backedge, true);
    }
    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
    return function;
}

llvm::Function* emit_reduction(llvm::Module& module, Op op, NodeId id, std::size_t count) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* f32 = builder.getFloatTy();
    auto* type = llvm::FunctionType::get(f32, {builder.getPtrTy()}, false);
    auto* function = llvm::Function::Create(type, llvm::Function::InternalLinkage,
        std::string("forge_reduce_") + op_name(op) + '_' + std::to_string(id), module);
    function->addFnAttr(llvm::Attribute::NoInline);
    auto* input = function->getArg(0);
    input->addAttr(llvm::Attribute::ReadOnly);

    auto* entry = llvm::BasicBlock::Create(context, "entry", function);
    builder.SetInsertPoint(entry);
    auto* zero = llvm::ConstantFP::get(f32, 0.0);
    if (count == 0) {
        builder.CreateRet(op == Op::reduce_mean ? llvm::ConstantFP::getNaN(f32) : zero);
        return function;
    }

    auto* loop = llvm::BasicBlock::Create(context, "ordered", function);
    auto* exit = llvm::BasicBlock::Create(context, "exit", function);
    builder.CreateBr(loop);
    builder.SetInsertPoint(loop);
    auto* index = builder.CreatePHI(builder.getInt64Ty(), 2, "index");
    index->addIncoming(builder.getInt64(0), entry);
    auto* accumulator = builder.CreatePHI(f32, 2, "accumulator");
    const bool maximum = op == Op::reduce_max;
    accumulator->addIncoming(maximum ? llvm::ConstantFP::getInfinity(f32, true) : zero, entry);
    auto* address = builder.CreateInBoundsGEP(f32, input, index);
    auto* value = builder.CreateAlignedLoad(f32, address, llvm::Align(1));
    llvm::Value* next_value = maximum
        ? builder.CreateIntrinsic(llvm::Intrinsic::maximum, {f32}, {accumulator, value})
        : builder.CreateFAdd(accumulator, value, "total");
    accumulator->addIncoming(next_value, loop);
    auto* next = builder.CreateNUWAdd(index, builder.getInt64(1), "next");
    index->addIncoming(next, loop);
    auto* length = builder.getInt64(count);
    set_loop_options(builder.CreateCondBr(builder.CreateICmpEQ(next, length), exit, loop), false);

    builder.SetInsertPoint(exit);
    if (op == Op::reduce_mean) {
        auto* denominator = builder.CreateUIToFP(length, f32);
        next_value = builder.CreateFDiv(next_value, denominator, "mean");
    }
    builder.CreateRet(next_value);
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
        } else if (node.shape.empty() && !is_reduction(node.op)) {
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
            if (is_reduction(node.op)) {
                const auto length = element_count(nodes[node.operands[0]].shape);
                auto* reduction = emit_reduction(module, node.op, id, length);
                values[id] = builder.CreateCall(reduction, arguments);
            } else {
                if (node.op == Op::matmul) {
                    throw std::invalid_argument("Matrix multiplication JIT lowering is unavailable");
                }
                auto* loop = emit_loop(module, graph, *next_region, arguments, simd);
                auto* output = pointer_at(buffers, plan.slots[id]);
                llvm::Value* length = node.shape == plan.output_shape
                    ? static_cast<llvm::Value*>(count) : builder.getInt64(element_count(node.shape));
                llvm::SmallVector<llvm::Value*, 6> call_arguments{output, length};
                call_arguments.append(arguments.begin(), arguments.end());
                builder.CreateCall(loop, call_arguments);
                values[id] = output;
            }
            ++next_region;
        }
    }
    if (plan.output_shape.empty()) {
        builder.CreateAlignedStore(values[graph.output()], pointer_at(buffers, 0), llvm::Align(1));
    }
    builder.CreateRetVoid();
}

} // namespace forge::detail
