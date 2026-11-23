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
    }
    throw std::logic_error("Unknown graph operation");
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

    // A zero dimension means the entire buffer is empty.
    if (std::ranges::find(shape, 0) == shape.end()) {
        std::size_t bytes = sizeof(float);
        for (const auto extent : shape) {
            if (extent > std::numeric_limits<std::size_t>::max() / bytes) {
                throw std::overflow_error("Input byte size exceeds the platform limit");
            }
            bytes *= extent;
        }
    }

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

Graph GraphBuilder::finish(NodeId output) && {
    if (nodes_.at(output).shape.empty()) {
        throw std::invalid_argument("The expression must produce an array");
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
