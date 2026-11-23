#include "kernel_plan.h"

#include <locale>
#include <sstream>
#include <stdexcept>

namespace forge::detail {
namespace {

bool is_pointwise_tensor(const Node& node) {
    return !node.shape.empty() &&
           (node.op == Op::add || node.op == Op::multiply || node.op == Op::sqrt);
}

std::string describe(const Graph& graph, const KernelPlan& plan) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << "loops=" << plan.loop_count() << " temporary_arrays=" << plan.temporary_arrays() << '\n';
    for (const auto& region : plan.regions) {
        text << "loop %" << region.output << " -> ";
        if (region.output == graph.output()) {
            text << "output\n";
        } else if (is_reduction(graph.nodes()[region.output].op)) {
            text << "scalar\n";
        } else {
            text << "scratch[" << plan.slots[region.output] << "]\n";
        }
        if (region.operations.empty()) {
            text << "  copy(%" << region.arguments.front() << ")\n";
        }
        for (const auto id : region.operations) {
            const auto& node = graph.nodes()[id];
            text << "  %" << id << " = " << op_name(node.op) << '(';
            for (std::size_t i = 0; i < node.operands.size(); ++i) {
                if (i != 0) {
                    text << ", ";
                }
                text << '%' << node.operands[i];
            }
            text << ")\n";
        }
    }
    return text.str();
}

} // namespace

KernelPlan plan_kernel(const Graph& graph, std::size_t input_count, FusionMode fusion) {
    const auto& nodes = graph.nodes();
    const auto none = nodes.size();
    KernelPlan plan;
    plan.used.resize(nodes.size());
    plan.slots.resize(nodes.size());
    plan.input_shapes.resize(input_count);
    plan.output_shape = nodes[graph.output()].shape;
    plan.slot_elements.push_back(element_count(plan.output_shape));
    plan.used[graph.output()] = true;

    std::vector<std::size_t> uses(nodes.size());
    std::vector<NodeId> consumers(nodes.size(), none);
    for (std::size_t i = nodes.size(); i > 0; --i) {
        const auto id = i - 1;
        if (!plan.used[id]) {
            continue;
        }
        const auto& node = nodes[id];
        for (const auto operand : node.operands) {
            plan.used[operand] = true;
            ++uses[operand];
            consumers[operand] = id;
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

    // Consumers have larger IDs than producers, so their region is known on this pass.
    // Count operand edges, including repeated edges to the same consumer.
    std::vector<NodeId> roots(nodes.size(), none);
    for (std::size_t i = nodes.size(); i > 0; --i) {
        const auto id = i - 1;
        if (!plan.used[id] || !is_pointwise_tensor(nodes[id])) {
            continue;
        }
        const auto consumer = consumers[id];
        const bool merge = fusion == FusionMode::enabled && uses[id] == 1 &&
                           is_pointwise_tensor(nodes[consumer]) &&
                           nodes[id].shape == nodes[consumer].shape;
        roots[id] = merge ? roots[consumer] : id;
    }

    // Order regions by their roots so shared producers are materialized before their users.
    std::vector<std::size_t> region_indices(nodes.size(), none);
    for (NodeId id = 0; id < nodes.size(); ++id) {
        const auto op = nodes[id].op;
        if (roots[id] == id || (plan.used[id] && (is_reduction(op) || op == Op::matmul))) {
            region_indices[id] = plan.regions.size();
            plan.regions.push_back({id, {}, {}});
            if (id != graph.output() && !is_reduction(op)) {
                plan.slots[id] = plan.slot_elements.size();
                plan.slot_elements.push_back(element_count(nodes[id].shape));
            }
        }
    }
    for (NodeId id = 0; id < nodes.size(); ++id) {
        if (roots[id] != none) {
            plan.regions[region_indices[roots[id]]].operations.push_back(id);
        } else if (region_indices[id] != none) {
            plan.regions[region_indices[id]].operations.push_back(id);
        }
    }
    std::vector<NodeId> argument_regions(nodes.size(), none);
    for (auto& region : plan.regions) {
        for (const auto id : region.operations) {
            for (const auto operand : nodes[id].operands) {
                if (roots[operand] != region.output && argument_regions[operand] != region.output) {
                    region.arguments.push_back(operand);
                    argument_regions[operand] = region.output;
                }
            }
        }
    }
    const auto output_op = nodes[graph.output()].op;
    if (output_op == Op::input || output_op == Op::splat) {
        plan.regions.push_back({graph.output(), {}, {graph.output()}});
    }
    plan.description = describe(graph, plan);
    return plan;
}

} // namespace forge::detail
