#include "llvm_lowering.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace forge::detail {
namespace {

const char* loop_name(Op op) {
    switch (op) {
    case Op::input: return "copy";
    case Op::add: return "add";
    case Op::multiply: return "multiply";
    case Op::sqrt: return "sqrt";
    default: throw std::logic_error("Expected a pointwise loop operation");
    }
}

llvm::Value* emit_value(llvm::IRBuilder<>& builder, Op op,
                        llvm::ArrayRef<llvm::Value*> operands) {
    switch (op) {
    case Op::input: return operands[0];
    case Op::add: return builder.CreateFAdd(operands[0], operands[1], "sum");
    case Op::multiply: return builder.CreateFMul(operands[0], operands[1], "product");
    case Op::sqrt:
        return builder.CreateIntrinsic(llvm::Intrinsic::sqrt, {builder.getFloatTy()}, operands);
    default: throw std::logic_error("Expected a pointwise arithmetic operation");
    }
}

llvm::Function* emit_loop(llvm::Module& module, Op op, NodeId id,
                          llvm::ArrayRef<llvm::Value*> operands) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* index_type = builder.getInt64Ty();
    auto* float_type = builder.getFloatTy();
    std::vector<llvm::Type*> parameters{builder.getPtrTy(), index_type};
    for (auto* operand : operands) {
        parameters.push_back(operand->getType());
    }

    auto* type = llvm::FunctionType::get(builder.getVoidTy(), parameters, false);
    auto* function = llvm::Function::Create(type, llvm::Function::InternalLinkage,
        std::string("forge_") + loop_name(op) + '_' + std::to_string(id), module);
    // Keep the unfused loop boundaries visible through LLVM's pipeline.
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
    std::vector<llvm::Value*> values;
    for (unsigned arg = 2; arg < function->arg_size(); ++arg) {
        auto* value = function->getArg(arg);
        value->setName("operand" + std::to_string(arg - 2));
        if (value->getType()->isPointerTy()) {
            value->addAttr(llvm::Attribute::ReadOnly);
            auto* address = builder.CreateInBoundsGEP(float_type, value, index);
            values.push_back(builder.CreateAlignedLoad(float_type, address, llvm::Align(1)));
        } else {
            values.push_back(value);
        }
    }
    auto* address = builder.CreateInBoundsGEP(float_type, output, index);
    builder.CreateAlignedStore(emit_value(builder, op, values), address, llvm::Align(1));
    auto* next = builder.CreateNUWAdd(index, builder.getInt64(1), "next");
    index->addIncoming(next, loop);
    builder.CreateCondBr(builder.CreateICmpEQ(next, count), exit, loop);
    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
    return function;
}

} // namespace

KernelPlan plan_kernel(const Graph& graph, std::size_t input_count) {
    const auto& nodes = graph.nodes();
    KernelPlan plan;
    plan.used.resize(nodes.size());
    plan.slots.resize(nodes.size());
    plan.input_shapes.resize(input_count);
    plan.output_shape = nodes[graph.output()].shape;
    plan.used[graph.output()] = true;

    for (std::size_t i = nodes.size(); i > 0; --i) {
        const auto id = i - 1;
        if (!plan.used[id]) {
            continue;
        }
        const auto& node = nodes[id];
        for (const auto operand : node.operands) {
            plan.used[operand] = true;
        }
        if (node.op == Op::input) {
            if (node.input_index >= input_count) {
                throw std::invalid_argument("Graph input index exceeds the argument count");
            }
            auto& shape = plan.input_shapes[node.input_index];
            if (shape && *shape != node.shape) {
                throw std::invalid_argument("Graph input has inconsistent shapes");
            }
            shape = node.shape;
        }
    }
    for (NodeId id = 0; id < nodes.size(); ++id) {
        const auto& node = nodes[id];
        if (plan.used[id] && id != graph.output() && !node.shape.empty() &&
            node.op != Op::input && node.op != Op::splat) {
            plan.slots[id] = plan.slot_count++;
        }
    }
    return plan;
}

void lower_kernel(llvm::Module& module, const Graph& graph, const KernelPlan& plan) {
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
        } else {
            std::vector<llvm::Value*> operands;
            for (const auto operand : node.operands) {
                operands.push_back(values[operand]);
            }
            if (node.shape.empty()) {
                values[id] = emit_value(builder, node.op, operands);
            } else {
                auto* loop = emit_loop(module, node.op, id, operands);
                auto* output = pointer_at(buffers, plan.slots[id]);
                std::vector<llvm::Value*> args{output, count};
                args.insert(args.end(), operands.begin(), operands.end());
                builder.CreateCall(loop, args);
                values[id] = output;
            }
        }
    }

    // Returning an input still gives the caller an independently owned array.
    if (nodes[graph.output()].op == Op::input) {
        auto* input = values[graph.output()];
        auto* loop = emit_loop(module, Op::input, graph.output(), {input});
        builder.CreateCall(loop, {pointer_at(buffers, 0), count, input});
    }
    builder.CreateRetVoid();
}

} // namespace forge::detail
