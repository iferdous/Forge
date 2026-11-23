#pragma once

#include "graph.h"

#include <optional>

namespace llvm {
class Module;
}

namespace forge::detail {

struct KernelPlan {
    std::vector<bool> used;
    std::vector<std::size_t> slots;
    std::vector<std::optional<Shape>> input_shapes;
    Shape output_shape;
    std::size_t slot_count = 1;  // Slot zero is the caller's output.
};

[[nodiscard]] KernelPlan plan_kernel(const Graph& graph, std::size_t input_count);
void lower_kernel(llvm::Module& module, const Graph& graph, const KernelPlan& plan);

} // namespace forge::detail
