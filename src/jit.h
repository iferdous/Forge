#pragma once

#include "graph.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace forge {

using InputBuffer = std::span<const std::byte>;
using OutputBuffer = std::span<std::byte>;

enum class KernelMode { scalar, avx2 };
[[nodiscard]] bool host_has_avx2();

struct KernelInfo {
    std::string triple;
    std::string data_layout;
    std::string cpu;
    std::string features;
    std::string ir;
    std::vector<std::string> remarks;
};

struct Assembly {
    KernelInfo info;
    std::string text;
};

class Workspace {
    friend class CompiledKernel;
    Workspace(std::size_t inputs, std::size_t slots, std::size_t scratch_elements);

    std::vector<float> scratch_;
    std::vector<const void*> inputs_;
    std::vector<void*> buffers_;
};

class CompiledKernel {
public:
    CompiledKernel(const Graph& graph, std::size_t input_count,
                   KernelMode mode = KernelMode::scalar);
    ~CompiledKernel();

    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;

    [[nodiscard]] const Shape& output_shape() const;
    [[nodiscard]] const std::vector<std::optional<Shape>>& input_shapes() const;
    [[nodiscard]] const KernelInfo& info() const;
    [[nodiscard]] const std::string& object() const;
    [[nodiscard]] Workspace make_workspace() const;

    // Separate workspaces let callers reuse the same compiled code concurrently.
    void run(std::span<const InputBuffer> inputs, OutputBuffer output, Workspace& workspace) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Emit Linux x86-64 assembly without linking or executing the foreign object.
[[nodiscard]] Assembly emit_x86_linux(const Graph& graph, std::size_t input_count,
                                     KernelMode mode = KernelMode::scalar);

} // namespace forge
