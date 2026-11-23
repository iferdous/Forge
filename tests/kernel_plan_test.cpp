#include "kernel_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>

namespace forge::detail {
namespace {

TEST(KernelPlan, CombinesSingleUsePointwiseNodesInOrder) {
    GraphBuilder builder;
    const auto input = builder.input(0, {9}, {});
    const auto two = builder.literal(2.0F, {});
    const auto add = builder.binary(Op::add, input, two, {});
    const auto three = builder.literal(3.0F, {});
    const auto multiply = builder.binary(Op::multiply, add, three, {});
    const auto output = builder.sqrt(multiply, {});
    const auto graph = std::move(builder).finish(output);

    const auto unfused = plan_kernel(graph, 1, FusionMode::disabled);
    ASSERT_EQ(unfused.loop_count(), 3U);
    EXPECT_EQ(unfused.temporary_arrays(), 2U);
    EXPECT_EQ(unfused.regions[0].operations, (std::vector{add}));
    EXPECT_EQ(unfused.regions[1].operations, (std::vector{multiply}));

    const auto fused = plan_kernel(graph, 1, FusionMode::enabled);
    ASSERT_EQ(fused.loop_count(), 1U);
    EXPECT_EQ(fused.temporary_arrays(), 0U);
    EXPECT_EQ(fused.regions[0].operations, (std::vector{add, multiply, output}));
    EXPECT_EQ(fused.regions[0].arguments.size(), 3U);
    EXPECT_EQ(fused.dump(),
              "loops=1 temporary_arrays=0\n"
              "loop %7 -> output\n"
              "  %3 = add(%0, %2)\n"
              "  %6 = multiply(%3, %5)\n"
              "  %7 = sqrt(%6)\n");
}

TEST(KernelPlan, PreservesLiveSharedProducerAndDeduplicatesItsReads) {
    GraphBuilder builder;
    const auto x = builder.input(0, {9}, {});
    const auto two = builder.literal(2.0F, {});
    const auto shared = builder.binary(Op::add, x, two, {});
    const auto squared = builder.binary(Op::multiply, shared, shared, {});
    const auto output = builder.sqrt(squared, {});
    const auto graph = std::move(builder).finish(output);

    const auto plan = plan_kernel(graph, 1, FusionMode::enabled);
    ASSERT_EQ(plan.loop_count(), 2U);
    EXPECT_EQ(plan.temporary_arrays(), 1U);
    EXPECT_EQ(plan.regions[0].operations, (std::vector{shared}));
    EXPECT_EQ(plan.regions[1].operations, (std::vector{squared, output}));
    EXPECT_EQ(plan.regions[1].arguments, (std::vector{shared}));
}

TEST(KernelPlan, IgnoresDeadConsumersAndSharesInputAcrossFusedBranches) {
    GraphBuilder builder;
    const auto x = builder.input(0, {2, 3}, {});
    const auto y = builder.input(1, {2, 3}, {});
    const auto left = builder.binary(Op::add, x, y, {});
    const auto right = builder.binary(Op::multiply, x, y, {});
    const auto output = builder.binary(Op::add, left, right, {});
    builder.sqrt(left, {});  // This branch is unreachable from the selected output.
    const auto graph = std::move(builder).finish(output);

    const auto plan = plan_kernel(graph, 2, FusionMode::enabled);
    ASSERT_EQ(plan.loop_count(), 1U);
    EXPECT_EQ(plan.regions[0].operations, (std::vector{left, right, output}));
    EXPECT_EQ(plan.regions[0].arguments, (std::vector{x, y}));
    EXPECT_EQ(plan.output_shape, (Shape{2, 3}));
}

TEST(KernelPlan, DeduplicatesArgumentsAcrossInterleavedRegions) {
    GraphBuilder builder;
    const auto x = builder.input(0, {9}, {});
    const auto two = builder.literal(2.0F, {});
    const auto early = builder.binary(Op::add, x, two, {});
    const auto shared = builder.binary(Op::multiply, x, x, {});
    const auto later = builder.binary(Op::add, early, x, {});
    const auto square = builder.binary(Op::multiply, shared, shared, {});
    const auto output = builder.binary(Op::add, later, square, {});
    const auto graph = std::move(builder).finish(output);

    const auto plan = plan_kernel(graph, 1, FusionMode::enabled);
    ASSERT_EQ(plan.loop_count(), 2U);
    EXPECT_EQ(plan.regions[0].output, shared);
    EXPECT_EQ(plan.regions[1].operations, (std::vector{early, later, square, output}));
    EXPECT_EQ(std::ranges::count(plan.regions[1].arguments, x), 1);
}

TEST(KernelPlan, CopiesIdentityAndKeepsScalarSplatOutsideLoop) {
    GraphBuilder identity;
    const auto x = identity.input(0, {0}, {});
    const auto graph = std::move(identity).finish(x);
    const auto plan = plan_kernel(graph, 1, FusionMode::enabled);
    ASSERT_EQ(plan.loop_count(), 1U);
    EXPECT_TRUE(plan.regions[0].operations.empty());
    EXPECT_EQ(plan.regions[0].arguments, (std::vector{x}));

    GraphBuilder with_scalar;
    const auto y = with_scalar.input(0, {9}, {});
    const auto four = with_scalar.literal(4.0F, {});
    const auto root = with_scalar.sqrt(four, {});
    const auto added = with_scalar.binary(Op::add, y, root, {});
    const auto scalar_graph = std::move(with_scalar).finish(added);
    const auto scalar_plan = plan_kernel(scalar_graph, 1, FusionMode::enabled);
    ASSERT_EQ(scalar_plan.loop_count(), 1U);
    EXPECT_EQ(scalar_plan.regions[0].operations, (std::vector{added}));
    ASSERT_EQ(scalar_plan.regions[0].arguments.size(), 2U);
    EXPECT_EQ(scalar_graph.nodes()[scalar_plan.regions[0].arguments[1]].op, Op::splat);
}

} // namespace
} // namespace forge::detail
