#pragma once

#include "graph.h"

#include <optional>

namespace forge {

enum class FusionMode { disabled, enabled };

namespace detail {

struct LoopRegion {
    NodeId output;
    std::vector<NodeId> operations;
    std::vector<NodeId> arguments;
};

struct KernelPlan {
    std::vector<bool> used;
    std::vector<std::size_t> slots;
    std::vector<std::optional<Shape>> input_shapes;
    Shape output_shape;
    std::vector<LoopRegion> regions;
    std::size_t slot_count = 1;  // Slot zero is the caller's output.
    std::string description;

    [[nodiscard]] std::size_t loop_count() const { return regions.size(); }
    [[nodiscard]] std::size_t temporary_arrays() const { return slot_count - 1; }
    [[nodiscard]] const std::string& dump() const { return description; }
};

[[nodiscard]] KernelPlan plan_kernel(const Graph& graph, std::size_t input_count, FusionMode fusion);

} // namespace detail
} // namespace forge
