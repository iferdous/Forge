#include "graph.h"
#include "jit.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace forge {
namespace {

Graph demo_graph(std::size_t size) {
    GraphBuilder builder;
    auto x = builder.input(0, {size}, {});
    auto two = builder.literal(2.0F, {});
    auto add = builder.binary(Op::add, x, two, {});
    auto three = builder.literal(3.0F, {});
    auto multiply = builder.binary(Op::multiply, add, three, {});
    auto output = builder.sqrt(multiply, {});
    return std::move(builder).finish(output);
}

TEST(ScalarJit, ExecutesTheDemoAcrossShapeBoundaries) {
    constexpr std::array source{1.0F, 10.0F, 25.0F, 46.0F};
    constexpr std::array expected{3.0F, 6.0F, 9.0F, 12.0F};
    for (const auto size : {0U, 1U, 9U, 1000U}) {
        SCOPED_TRACE(size);
        CompiledKernel kernel(demo_graph(size), 1);
        auto workspace = kernel.make_workspace();
        std::vector<float> input(size);
        std::vector<float> output(size, -1.0F);
        for (std::size_t i = 0; i < size; ++i) {
            input[i] = source[i % source.size()];
        }
        const std::array inputs{std::as_bytes(std::span(input))};
        kernel.run(inputs, std::as_writable_bytes(std::span(output)), workspace);
        for (std::size_t i = 0; i < size; ++i) {
            EXPECT_EQ(output[i], expected[i % expected.size()]);
        }
    }
}

TEST(ScalarJit, ProvidesUnfusedScalarIrAndCrossAssembly) {
    const auto graph = demo_graph(9);
    CompiledKernel kernel(graph, 1, KernelMode::scalar, FusionMode::disabled);
    EXPECT_EQ(kernel.plan().loop_count(), 3U);
    EXPECT_NE(kernel.info().ir.find("fadd float"), std::string::npos);
    EXPECT_NE(kernel.info().ir.find("fmul float"), std::string::npos);
    EXPECT_NE(kernel.info().ir.find("llvm.sqrt.f32"), std::string::npos);
    EXPECT_EQ(kernel.info().ir.find("<8 x float>"), std::string::npos);
    EXPECT_FALSE(kernel.object().empty());

    const auto emitted = emit_x86_linux(graph, 1);
    EXPECT_EQ(emitted.info.triple, "x86_64-unknown-linux-gnu");
    EXPECT_NE(emitted.text.find("sqrtss"), std::string::npos);
}

TEST(ScalarJit, RejectsBadBuffersAndOverlappingOutput) {
    CompiledKernel kernel(demo_graph(2), 1);
    auto workspace = kernel.make_workspace();
    std::array values{1.0F, 10.0F, 0.0F};
    const std::array inputs{InputBuffer(std::as_bytes(std::span(values).first<2>()))};
    const auto overlap = std::as_writable_bytes(std::span(values).last<2>());
    const auto wrong_size = std::as_writable_bytes(std::span(values));
    EXPECT_THROW(kernel.run(inputs, overlap, workspace), std::invalid_argument);
    EXPECT_THROW(kernel.run(inputs, wrong_size, workspace), std::invalid_argument);
    EXPECT_THROW(CompiledKernel(demo_graph(2), 0), std::invalid_argument);

    std::array<float, 2> output{};
    kernel.run(inputs, std::as_writable_bytes(std::span(output)), workspace);
    EXPECT_EQ(output, (std::array{3.0F, 6.0F}));
}

TEST(MatmulJit, MatchesScalarReferenceAcrossTileBoundaries) {
    constexpr std::size_t m = 17, k = 35, n = 33;
    GraphBuilder builder;
    const auto a = builder.input(0, {m, k}, {});
    const auto b = builder.input(1, {k, n}, {});
    const auto product = builder.matmul(a, b, {});
    const auto graph = std::move(builder).finish(product);
    std::vector<float> left(m * k);
    std::vector<float> right(k * n);
    std::vector<float> output(m * n);
    std::vector<float> reference(m * n);
    for (std::size_t i = 0; i < left.size(); ++i) {
        left[i] = static_cast<float>(i % 17) * 0.83F - 7.0F;
    }
    for (std::size_t i = 0; i < right.size(); ++i) {
        right[i] = static_cast<float>(i % 19) * 0.19F - 3.0F;
    }
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < k; ++inner) {
                // A separate rounded product prevents contraction in the reference.
                const volatile float product = left[row * k + inner] * right[inner * n + col];
                sum += product;
            }
            reference[row * n + col] = sum;
        }
    }

    const std::array inputs{std::as_bytes(std::span(left)), std::as_bytes(std::span(right))};
    for (const auto mode : {KernelMode::scalar, KernelMode::avx2}) {
        if (mode == KernelMode::avx2 && !host_has_avx2()) {
            continue;
        }
        CompiledKernel kernel(graph, 2, mode);
        auto workspace = kernel.make_workspace();
        kernel.run(inputs, std::as_writable_bytes(std::span(output)), workspace);
        for (std::size_t i = 0; i < output.size(); ++i) {
            ASSERT_EQ(std::bit_cast<std::uint32_t>(output[i]), std::bit_cast<std::uint32_t>(reference[i]))
                << "element " << i;
        }
    }
}

TEST(ReductionJit, RejectsOverflowAcrossDifferentlySizedWorkspaceSlots) {
    GraphBuilder builder;
    const auto size = std::vector<float>().max_size() / 2 + 1;
    const auto input = builder.input(0, {size}, {});
    const auto small = builder.input(1, {1}, {});
    const auto two = builder.literal(2.0F, {});
    const auto large_add = builder.binary(Op::add, input, two, {});
    const auto large_multiply = builder.binary(Op::multiply, large_add, two, {});
    const auto large_sum = builder.reduce(Op::reduce_sum, large_multiply, {});
    const auto output = builder.binary(Op::add, small, large_sum, {});
    const auto graph = std::move(builder).finish(output);
    EXPECT_THROW(CompiledKernel(graph, 2, KernelMode::scalar, FusionMode::disabled), std::overflow_error);
}

} // namespace
} // namespace forge
