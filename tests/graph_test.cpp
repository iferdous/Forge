#include "graph.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace forge {

TEST(GraphTest, InsertsExplicitSplatsAndTracksSharedUses) {
    GraphBuilder builder;
    const auto input = builder.input(0, {4}, {1, 1});
    const auto constant = builder.literal(-0.0F, {1, 5});
    const auto shared = builder.binary(Op::add, input, constant, {1, 1});
    const auto output = builder.binary(Op::multiply, shared, shared, {1, 1});
    const auto graph = std::move(builder).finish(output);

    ASSERT_EQ(graph.nodes().size(), 5U);
    EXPECT_EQ(graph.nodes()[shared].use_count, 2U);
    EXPECT_EQ(graph.dump(),
              "%0 = input(0) : f32[4] @1:1 uses=1\n"
              "%1 = literal(0x80000000) : f32 @1:5 uses=1\n"
              "%2 = splat(%1) : f32[4] @1:5 uses=1\n"
              "%3 = add(%0, %2) : f32[4] @1:1 uses=2\n"
              "%4 = multiply(%3, %3) : f32[4] @1:1 uses=0\n"
              "return %4\n");
}

TEST(GraphTest, PreservesOperandOrderWithScalarOnTheLeft) {
    GraphBuilder builder;
    const auto scalar = builder.sqrt(builder.literal(4.0F, {}), {});
    const auto array = builder.input(1, {2, 0}, {});
    const auto output = builder.binary(Op::multiply, scalar, array, {});
    const auto graph = std::move(builder).finish(output);

    const auto& operands = graph.nodes()[output].operands;
    ASSERT_EQ(operands.size(), 2U);
    EXPECT_EQ(graph.nodes()[operands[0]].op, Op::splat);
    EXPECT_EQ(graph.nodes()[operands[0]].operands, std::vector<NodeId>{scalar});
    EXPECT_EQ(operands[1], array);
    EXPECT_EQ(graph.nodes()[output].shape, (Shape{2, 0}));
}

TEST(GraphTest, RejectsInvalidGraphOperations) {
    GraphBuilder builder;
    const auto left = builder.input(0, {2, 1}, {});
    const auto right = builder.input(1, {2}, {});
    const auto scalar = builder.literal(1.0F, {});
    EXPECT_THROW(builder.binary(Op::add, left, right, {}), std::invalid_argument);
    EXPECT_THROW(builder.binary(Op::sqrt, left, left, {}), std::invalid_argument);
    EXPECT_THROW(builder.sqrt(99, {}), std::out_of_range);
    EXPECT_THROW(static_cast<void>(std::move(builder).finish(scalar)), std::invalid_argument);
    EXPECT_EQ(std::move(builder).finish(left).output(), left);
}

TEST(GraphTest, ChecksInputRankAndByteSizeEvenForLargeShapes) {
    GraphBuilder builder;
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(builder.input(0, {}, {}), std::invalid_argument);
    EXPECT_THROW(builder.input(0, {1, 1, 1}, {}), std::invalid_argument);
    EXPECT_THROW(builder.input(0, {maximum, 2}, {}), std::overflow_error);
    EXPECT_THROW(builder.input(0, {maximum / sizeof(float) + 1}, {}), std::overflow_error);
    EXPECT_THROW(builder.literal(std::numeric_limits<float>::infinity(), {}),
                 std::invalid_argument);
    const auto input = builder.input(0, {maximum, 0}, {});
    EXPECT_EQ(std::move(builder).finish(input).nodes()[input].shape, (Shape{maximum, 0}));
}

} // namespace forge
