#include "jit.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <string_view>
#include <vector>

namespace py = pybind11;
using namespace forge;

namespace {

KernelMode kernel_mode(std::string_view target) {
    if (target == "scalar") {
        return KernelMode::scalar;
    }
    if (target == "avx2") {
        return KernelMode::avx2;
    }
    throw py::value_error("Kernel target must be 'scalar' or 'avx2'");
}

py::array checked_array(py::handle value) {
    if (!py::isinstance<py::array>(value)) {
        throw py::type_error("Expected a native-endian float32 NumPy array");
    }
    auto array = py::reinterpret_borrow<py::array>(value);
    if (!array.dtype().equal(py::dtype::of<float>())) {
        throw py::type_error("Expected a native-endian float32 NumPy array");
    }
    if (array.ndim() < 1 || array.ndim() > 2 || !(array.flags() & py::array::c_style)) {
        throw py::value_error("Expected a C-contiguous array of rank one or two");
    }
    return array;
}

void check_shape(const py::array& array, const Shape& expected) {
    if (static_cast<std::size_t>(array.ndim()) != expected.size()) {
        throw py::value_error("Array shape does not match the compiled shape");
    }
    for (std::size_t axis = 0; axis < expected.size(); ++axis) {
        if (static_cast<std::size_t>(array.shape(axis)) != expected[axis]) {
            throw py::value_error("Array shape does not match the compiled shape");
        }
    }
}

class PythonKernel {
public:
    PythonKernel(const Graph& graph, std::size_t input_count, std::string_view target)
        : kernel_(graph, input_count, kernel_mode(target)), workspace_(kernel_.make_workspace()) {}

    py::array operator()(const py::args& arguments) {
        std::vector<py::ssize_t> shape(kernel_.output_shape().begin(), kernel_.output_shape().end());
        py::array_t<float> output(shape);
        execute(output, arguments);
        return output;
    }

    void run_into(py::handle output, const py::args& arguments) {
        execute(checked_array(output), arguments);
    }

    const KernelInfo& info() const { return kernel_.info(); }
    py::bytes object() const { return py::bytes(kernel_.object()); }

private:
    void execute(py::array output, const py::args& arguments) {
        const auto& shapes = kernel_.input_shapes();
        if (arguments.size() != shapes.size()) {
            throw py::type_error("Incorrect number of kernel input arrays");
        }
        check_shape(output, kernel_.output_shape());
        if (!output.writeable()) {
            throw py::value_error("Kernel output must be writable");
        }
        std::vector<InputBuffer> inputs;
        inputs.reserve(arguments.size());
        for (std::size_t index = 0; index < arguments.size(); ++index) {
            auto array = checked_array(arguments[index]);
            if (shapes[index]) {
                check_shape(array, *shapes[index]);
            }
            inputs.emplace_back(static_cast<const std::byte*>(array.data()), array.nbytes());
        }

        // The Python arguments and output own these buffers throughout the synchronous call.
        kernel_.run(inputs,
                    {static_cast<std::byte*>(output.mutable_data()),
                     static_cast<std::size_t>(output.nbytes())},
                    workspace_);
    }

    CompiledKernel kernel_;
    Workspace workspace_;
};

} // namespace

void bind_jit(py::module_& module) {
    py::class_<KernelInfo>(module, "KernelInfo")
        .def_readonly("triple", &KernelInfo::triple)
        .def_readonly("data_layout", &KernelInfo::data_layout)
        .def_readonly("cpu", &KernelInfo::cpu)
        .def_readonly("features", &KernelInfo::features)
        .def_readonly("ir", &KernelInfo::ir)
        .def_readonly("remarks", &KernelInfo::remarks);

    py::class_<Assembly>(module, "Assembly")
        .def_readonly("info", &Assembly::info)
        .def_readonly("text", &Assembly::text);

    py::class_<PythonKernel>(module, "CompiledKernel")
        .def(py::init<const Graph&, std::size_t, std::string_view>())
        .def("__call__", &PythonKernel::operator())
        .def("run_into", &PythonKernel::run_into, py::arg("output"))
        .def_property_readonly("info", &PythonKernel::info)
        .def_property_readonly("object", &PythonKernel::object);

    module.def("has_avx2", &host_has_avx2);
    module.def("emit_x86_linux", [](const Graph& graph, std::size_t inputs, std::string_view target) {
        return emit_x86_linux(graph, inputs, kernel_mode(target));
    });
}
