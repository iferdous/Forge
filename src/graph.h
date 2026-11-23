#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace forge {

using NodeId = std::size_t;
using Shape = std::vector<std::size_t>;

enum class Op { input, literal, splat, add, multiply, sqrt };

struct SourceLocation {
    std::size_t line = 1;
    std::size_t column = 1;
};

struct Node {
    Op op;
    SourceLocation source;
    Shape shape;
    std::vector<NodeId> operands;
    std::size_t input_index = 0;
    float literal_value = 0.0F;
    std::size_t use_count = 0;
};

class GraphBuilder;

class Graph {
public:
    [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }
    [[nodiscard]] NodeId output() const { return output_; }
    [[nodiscard]] std::string dump() const;

private:
    friend class GraphBuilder;
    Graph(std::vector<Node> nodes, NodeId output);

    std::vector<Node> nodes_;
    NodeId output_;
};

class GraphBuilder {
public:
    NodeId input(std::size_t index, Shape shape, SourceLocation source);
    NodeId literal(float value, SourceLocation source);
    NodeId binary(Op op, NodeId left, NodeId right, SourceLocation source);
    NodeId sqrt(NodeId operand, SourceLocation source);
    [[nodiscard]] Graph finish(NodeId output) &&;

private:
    NodeId append(Node node);
    std::vector<Node> nodes_;
};

} // namespace forge
