#include "jit.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <memory>
#include <optional>
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
    if (array.ndim() > 2 || !(array.flags() & py::array::c_style)) {
        throw py::value_error("Expected a C-contiguous float32 buffer of rank at most two");
    }
    // The IR uses byte-aligned loads and stores. Neither float-alignment nor
    // SIMD-alignment is required, and the view is borrowed without conversion.
    if (array.size() != 0 && array.data() == nullptr) {
        throw py::value_error("A nonempty array must have a data buffer");
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

std::shared_ptr<CompiledKernel> compile_code(const Graph& graph, std::size_t input_count,
                                             std::string_view target, bool fusion) {
    return std::make_shared<CompiledKernel>(graph, input_count, kernel_mode(target),
        fusion ? FusionMode::enabled : FusionMode::disabled);
}

class PythonKernel {
public:
    PythonKernel(const Graph& graph, std::size_t input_count, std::string_view target, bool fusion)
        : PythonKernel(compile_code(graph, input_count, target, fusion)) {}

    explicit PythonKernel(std::shared_ptr<CompiledKernel> code)
        : code_(std::move(code)), output_shape_(code_->output_shape().begin(), code_->output_shape().end()),
          inputs_(code_->input_shapes().size()) {}

    py::array operator()(const py::args& arguments) {
        bind_inputs(arguments);
        py::array_t<float> output(output_shape_);
        execute(output);
        return output;
    }

    void run_into(py::handle destination, const py::args& arguments) {
        auto output = checked_array(destination);
        check_shape(output, code_->output_shape());
        if (!output.writeable()) {
            throw py::value_error("Kernel output must be writable");
        }
        bind_inputs(arguments);
        execute(output);
    }

    const KernelInfo& info() const { return code_->info(); }
    const detail::KernelPlan& plan() const { return code_->plan(); }
    py::bytes object() const { return py::bytes(code_->object()); }

private:
    void bind_inputs(const py::args& arguments) {
        const auto& shapes = code_->input_shapes();
        if (arguments.size() != shapes.size()) {
            throw py::type_error("Incorrect number of kernel input arrays");
        }
        for (std::size_t index = 0; index < arguments.size(); ++index) {
            auto array = checked_array(arguments[index]);
            if (array.ndim() == 0) {
                throw py::value_error("Expected a C-contiguous input array of rank one or two");
            }
            if (shapes[index]) {
                check_shape(array, *shapes[index]);
            }
            inputs_[index] = {static_cast<const std::byte*>(array.data()),
                              static_cast<std::size_t>(array.nbytes())};
        }
    }

    void execute(py::array& output) {
        if (!workspace_) {
            workspace_.emplace(code_->make_workspace());
        }
        // The GIL serializes calls on one handle. The original Python arguments
        // and output keep their storage alive across synchronous execution.
        code_->run(inputs_, {static_cast<std::byte*>(output.mutable_data()),
                             static_cast<std::size_t>(output.nbytes())}, *workspace_);
    }

    std::shared_ptr<CompiledKernel> code_;
    std::vector<py::ssize_t> output_shape_;
    std::vector<InputBuffer> inputs_;
    std::optional<Workspace> workspace_;
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

    py::class_<CompiledKernel, std::shared_ptr<CompiledKernel>>(module, "_CompiledCode");
    module.def("_compile_code", &compile_code);

    py::class_<PythonKernel>(module, "CompiledKernel")
        .def(py::init<const Graph&, std::size_t, std::string_view, bool>())
        .def(py::init<std::shared_ptr<CompiledKernel>>(), py::arg("code").none(false))
        .def("__call__", &PythonKernel::operator())
        .def("run_into", &PythonKernel::run_into, py::arg("output"))
        .def_property_readonly("info", &PythonKernel::info)
        .def_property_readonly("plan", &PythonKernel::plan)
        .def_property_readonly("object", &PythonKernel::object);

    module.def("has_avx2", &host_has_avx2);
    module.def("emit_x86_linux", [](const Graph& graph, std::size_t inputs, std::string_view target, bool fusion) {
        return emit_x86_linux(graph, inputs, kernel_mode(target),
                              fusion ? FusionMode::enabled : FusionMode::disabled);
    });
}
