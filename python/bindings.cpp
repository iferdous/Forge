#include "kernel_plan.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;

#ifdef FORGE_ENABLE_JIT
void bind_jit(py::module_& module);
#endif

PYBIND11_MODULE(_forge_ir, module) {
    using namespace forge;

    py::enum_<Op>(module, "Op")
        .value("INPUT", Op::input)
        .value("LITERAL", Op::literal)
        .value("SPLAT", Op::splat)
        .value("ADD", Op::add)
        .value("MULTIPLY", Op::multiply)
        .value("SQRT", Op::sqrt)
        .value("SUM", Op::reduce_sum)
        .value("MAX", Op::reduce_max)
        .value("MEAN", Op::reduce_mean)
        .value("MATMUL", Op::matmul);

    py::class_<SourceLocation>(module, "SourceLocation")
        .def(py::init<std::size_t, std::size_t>())
        .def_readonly("line", &SourceLocation::line)
        .def_readonly("column", &SourceLocation::column);

    py::class_<Node>(module, "Node")
        .def_readonly("op", &Node::op)
        .def_readonly("shape", &Node::shape)
        .def_readonly("source", &Node::source)
        .def_readonly("operands", &Node::operands)
        .def_readonly("input_index", &Node::input_index)
        .def_readonly("literal_value", &Node::literal_value)
        .def_readonly("use_count", &Node::use_count);

    py::class_<Graph>(module, "Graph")
        .def_property_readonly("nodes", &Graph::nodes, py::return_value_policy::copy)
        .def_property_readonly("output", &Graph::output)
        .def("dump", &Graph::dump);

    py::class_<detail::KernelPlan>(module, "KernelPlan")
        .def_property_readonly("loop_count", &detail::KernelPlan::loop_count)
        .def_property_readonly("temporary_arrays", &detail::KernelPlan::temporary_arrays)
        .def("dump", &detail::KernelPlan::dump);

    module.def("plan_kernel", [](const Graph& graph, std::size_t inputs, bool fusion) {
        return detail::plan_kernel(graph, inputs, fusion ? FusionMode::enabled : FusionMode::disabled);
    });

    py::class_<GraphBuilder>(module, "GraphBuilder")
        .def(py::init<>())
        .def("input", &GraphBuilder::input)
        .def("literal", &GraphBuilder::literal)
        .def("binary", &GraphBuilder::binary)
        .def("sqrt", &GraphBuilder::sqrt)
        .def("reduce", &GraphBuilder::reduce)
        .def("matmul", &GraphBuilder::matmul)
        .def("finish", [](GraphBuilder& builder, NodeId output) {
            return std::move(builder).finish(output);
        });
#ifdef FORGE_ENABLE_JIT
    bind_jit(module);
#endif
}
