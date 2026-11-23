#pragma once

#include "kernel_plan.h"

namespace llvm {
class Module;
}

namespace forge::detail {

void lower_kernel(llvm::Module& module, const Graph& graph, const KernelPlan& plan, bool simd);

} // namespace forge::detail
