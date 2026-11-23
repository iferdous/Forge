#include "graph.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace forge {
namespace {

std::string shape_text(const Shape& shape) {
    if (shape.empty()) {
        return "f32";
    }

    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << "f32[" << shape[0];
    if (shape.size() == 2) {
        text << ", " << shape[1];
    }
    text << ']';
    return text.str();
}

} // namespace

const char* op_name(Op op) {
    switch (op) {
    case Op::input: return "input";
    case Op::literal: return "literal";
    case Op::splat: return "splat";
    case Op::add: return "add";
    case Op::multiply: return "multiply";
    case Op::sqrt: return "sqrt";
    case Op::reduce_sum: return "sum";
    case Op::reduce_max: return "max";
    case Op::reduce_mean: return "mean";
    case Op::matmul: return "matmul";
    }
    throw std::logic_error("Unknown graph operation");
}

bool is_reduction(Op op) {
    return op == Op::reduce_sum || op == Op::reduce_max || op == Op::reduce_mean;
}

std::size_t element_count(const Shape& shape) {
    if (std::ranges::find(shape, 0) != shape.end()) {
        return 0;
    }
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    std::size_t count = 1;
    for (const auto extent : shape) {
        if (extent > limit / count) {
            throw std::overflow_error("Array byte size exceeds the platform limit");
        }
        count *= extent;
    }
    return count;
}

Graph::Graph(std::vector<Node> nodes, NodeId output)
    : nodes_(std::move(nodes)), output_(output) {}

std::string Graph::dump() const {
    std::ostringstream text;
    text.imbue(std::locale::classic());

    for (NodeId id = 0; id < nodes_.size(); ++id) {
        const auto& node = nodes_[id];
        text << '%' << id << " = " << op_name(node.op) << '(';
        if (node.op == Op::input) {
            text << node.input_index;
        } else if (node.op == Op::literal) {
            text << "0x" << std::setfill('0') << std::setw(8) << std::hex
                 << std::bit_cast<std::uint32_t>(node.literal_value) << std::dec;
        } else {
            for (std::size_t index = 0; index < node.operands.size(); ++index) {
                if (index != 0) {
                    text << ", ";
                }
                text << '%' << node.operands[index];
            }
        }
        text << ") : " << shape_text(node.shape) << " @" << node.source.line << ':'
             << node.source.column << " uses=" << node.use_count << '\n';
    }

    text << "return %" << output_ << '\n';
    return text.str();
}

NodeId GraphBuilder::input(std::size_t index, Shape shape, SourceLocation source) {
    if (shape.empty() || shape.size() > 2) {
        throw std::invalid_argument("Input must have rank one or two");
    }

    static_cast<void>(element_count(shape));

    Node node{Op::input, source, std::move(shape), {}};
    node.input_index = index;
    return append(std::move(node));
}

NodeId GraphBuilder::literal(float value, SourceLocation source) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Literal must be finite as float32");
    }
    Node node{Op::literal, source, {}, {}};
    node.literal_value = value;
    return append(std::move(node));
}

NodeId GraphBuilder::binary(Op op, NodeId left, NodeId right, SourceLocation source) {
    if (op != Op::add && op != Op::multiply) {
        throw std::invalid_argument("Expected a binary elementwise operation");
    }
    const auto& left_shape = nodes_.at(left).shape;
    const auto& right_shape = nodes_.at(right).shape;
    if (!left_shape.empty() && !right_shape.empty() && left_shape != right_shape) {
        throw std::invalid_argument("Tensor shapes must match (" + shape_text(left_shape) +
                                    " and " + shape_text(right_shape) + ")");
    }

    Shape output_shape = left_shape.empty() ? right_shape : left_shape;
    // Explicit splats let later passes distinguish a scalar broadcast from an array input.
    if (left_shape.empty() && !right_shape.empty()) {
        left = append({Op::splat, nodes_[left].source, output_shape, {left}});
    } else if (!left_shape.empty() && right_shape.empty()) {
        right = append({Op::splat, nodes_[right].source, output_shape, {right}});
    }
    return append({op, source, std::move(output_shape), {left, right}});
}

NodeId GraphBuilder::sqrt(NodeId operand, SourceLocation source) {
    return append({Op::sqrt, source, nodes_.at(operand).shape, {operand}});
}

NodeId GraphBuilder::reduce(Op op, NodeId operand, SourceLocation source) {
    const auto& shape = nodes_.at(operand).shape;
    if (!is_reduction(op) || shape.empty()) {
        throw std::invalid_argument("Expected a reduction over an array");
    }
    if (op == Op::reduce_max && element_count(shape) == 0) {
        throw std::invalid_argument("max requires a nonempty array");
    }
    return append({op, source, {}, {operand}});
}

NodeId GraphBuilder::matmul(NodeId left, NodeId right, SourceLocation source) {
    const auto& a = nodes_.at(left).shape;
    const auto& b = nodes_.at(right).shape;
    if (a.size() != 2 || b.size() != 2 || a[1] != b[0]) {
        throw std::invalid_argument("Matrix multiplication requires rank-two arrays with equal inner dimensions");
    }
    Shape output{a[0], b[1]};
    static_cast<void>(element_count(output));
    return append({Op::matmul, source, std::move(output), {left, right}});
}

Graph GraphBuilder::finish(NodeId output) && {
    if (nodes_.at(output).shape.empty()) {
        // Constant-only scalar expressions are outside the array language.
        std::vector<bool> used(output + 1);
        used[output] = true;
        bool has_reduction = false;
        for (auto i = output + 1; i > 0 && !has_reduction; --i) {
            const auto id = i - 1;
            if (used[id]) {
                has_reduction = is_reduction(nodes_[id].op);
                for (const auto operand : nodes_[id].operands) {
                    used[operand] = true;
                }
            }
        }
        if (!has_reduction) {
            throw std::invalid_argument("The expression must produce an array or reduce one");
        }
    }

    for (const auto& node : nodes_) {
        for (const auto operand : node.operands) {
            ++nodes_[operand].use_count;
        }
    }

    return Graph(std::move(nodes_), output);
}

NodeId GraphBuilder::append(Node node) {
    const auto id = nodes_.size();
    nodes_.push_back(std::move(node));
    return id;
}

} // namespace forge
