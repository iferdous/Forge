#include "jit.h"
#include "llvm_lowering.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/IR/DiagnosticHandler.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Host.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace forge {
namespace {

using EntryPoint = void(std::size_t, const void* const*, void* const*);
static_assert(sizeof(std::size_t) == 8);

void initialize_targets() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        if (llvm::InitializeNativeTarget() || llvm::InitializeNativeTargetAsmPrinter()) {
            throw std::runtime_error("LLVM could not initialize the native target");
        }
        if (!llvm::Triple(llvm::sys::getProcessTriple()).isX86()) {
            LLVMInitializeX86TargetInfo();
            LLVMInitializeX86Target();
            LLVMInitializeX86TargetMC();
            LLVMInitializeX86AsmPrinter();
        }
    });
}

template <typename T>
T require(llvm::Expected<T> value, const char* action) {
    if (!value) {
        throw std::runtime_error(std::string(action) + ": " + llvm::toString(value.takeError()));
    }
    return std::move(*value);
}

void check(llvm::Error error, const char* action) {
    if (error) {
        throw std::runtime_error(std::string(action) + ": " + llvm::toString(std::move(error)));
    }
}

llvm::orc::JITTargetMachineBuilder target_builder(llvm::Triple triple, KernelMode mode) {
    llvm::orc::JITTargetMachineBuilder target(triple);
    target.setCPU(triple.isX86() ? "x86-64" : "generic");
    if (triple.isX86()) {
        target.setFeatures(mode == KernelMode::avx2
                           ? "+avx,+avx2,-avx512f,-fma" : "-avx,-avx2,-fma");
    }
    target.getOptions().AllowFPOpFusion = llvm::FPOpFusion::Strict;
    return target;
}

void verify(llvm::Module& module) {
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    if (llvm::verifyModule(module, &stream)) {
        throw std::runtime_error("LLVM module verification failed: " + diagnostic);
    }
}

class VectorizationDiagnostics final : public llvm::DiagnosticHandler {
public:
    bool isAnyRemarkEnabled() const override { return true; }
    bool isPassedOptRemarkEnabled(llvm::StringRef pass) const override {
        return pass == "loop-vectorize";
    }
    bool isMissedOptRemarkEnabled(llvm::StringRef pass) const override {
        return pass == "loop-vectorize";
    }
    bool handleDiagnostics(const llvm::DiagnosticInfo& diagnostic) override {
        const auto* remark = llvm::dyn_cast<llvm::DiagnosticInfoOptimizationBase>(&diagnostic);
        if (!remark || remark->getPassName() != "loop-vectorize") {
            return false;
        }

        std::string message;
        llvm::raw_string_ostream stream(message);
        stream << remark->getFunction().getName() << ' ' << remark->getRemarkName() << ": ";
        llvm::DiagnosticPrinterRawOStream printer(stream);
        remark->print(printer);
        remarks.push_back(std::move(message));
        return true;
    }

    std::vector<std::string> remarks;
};

std::unique_ptr<llvm::Module> build_module(const Graph& graph, const detail::KernelPlan& plan,
                                           llvm::LLVMContext& context, llvm::TargetMachine& machine,
                                           KernelMode mode, std::vector<std::string>& remarks) {
    auto module = std::make_unique<llvm::Module>("forge", context);
    module->setTargetTriple(machine.getTargetTriple());
    module->setDataLayout(machine.createDataLayout());
    const auto simd = mode == KernelMode::avx2;
    detail::lower_kernel(*module, graph, plan, simd);
    verify(*module);

    llvm::PipelineTuningOptions tuning;
    VectorizationDiagnostics* captured = nullptr;
    if (simd) {
        auto diagnostics = std::make_unique<VectorizationDiagnostics>();
        captured = diagnostics.get();
        context.setDiagnosticHandler(std::move(diagnostics), /*RespectFilters=*/true);
    }
    tuning.LoopVectorization = simd;
    tuning.SLPVectorization = false;
    tuning.LoopUnrolling = false;
    llvm::PassBuilder builder(&machine, tuning);
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager sccs;
    llvm::ModuleAnalysisManager modules;
    builder.registerLoopAnalyses(loops);
    builder.registerFunctionAnalyses(functions);
    builder.registerCGSCCAnalyses(sccs);
    builder.registerModuleAnalyses(modules);
    builder.crossRegisterProxies(loops, functions, sccs, modules);
    builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2).run(*module, modules);
    verify(*module);
    if (captured) {
        remarks = std::move(captured->remarks);
    }
    return module;
}

KernelInfo inspect(llvm::Module& module, const llvm::TargetMachine& machine,
                   std::vector<std::string> remarks) {
    KernelInfo info{module.getTargetTriple().str(), module.getDataLayoutStr(),
                    machine.getTargetCPU().str(), machine.getTargetFeatureString().str(),
                    {}, std::move(remarks)};
    llvm::raw_string_ostream stream(info.ir);
    module.print(stream, nullptr);
    return info;
}

std::size_t elements(const Shape& shape) {
    if (std::ranges::find(shape, 0) != shape.end()) {
        return 0;
    }
    std::size_t count = 1;
    for (const auto extent : shape) {
        count *= extent;  // Graph construction already checked array byte-size overflow.
    }
    return count;
}

bool overlaps(InputBuffer input, OutputBuffer output) {
    if (input.empty() || output.empty()) {
        return false;
    }
    const auto a = reinterpret_cast<std::uintptr_t>(input.data());
    const auto b = reinterpret_cast<std::uintptr_t>(output.data());
    return a <= b ? b - a < input.size() : a - b < output.size();
}

} // namespace

bool host_has_avx2() {
    return llvm::Triple(llvm::sys::getProcessTriple()).getArch() == llvm::Triple::x86_64 &&
           llvm::sys::getHostCPUFeatures().lookup("avx2");
}

struct CompiledKernel::Impl {
    detail::KernelPlan plan;
    std::size_t count = 0;
    std::size_t scratch_elements = 0;
    KernelInfo info;
    std::string object;
    std::unique_ptr<llvm::orc::LLJIT> jit;
    EntryPoint* entry = nullptr;
};

Workspace::Workspace(std::size_t inputs, std::size_t slots, std::size_t scratch_elements)
    : scratch_(scratch_elements), inputs_(inputs), buffers_(slots) {}

CompiledKernel::CompiledKernel(const Graph& graph, std::size_t input_count, KernelMode mode)
    : impl_(std::make_unique<Impl>()) {
    if (mode == KernelMode::avx2 && !host_has_avx2()) {
        throw std::invalid_argument("Native AVX2 requires an x86-64 process with usable AVX2");
    }
    initialize_targets();
    impl_->plan = detail::plan_kernel(graph, input_count);
    impl_->count = elements(impl_->plan.output_shape);
    const auto scratch_slots = impl_->plan.slot_count - 1;
    if (scratch_slots != 0 &&
        impl_->count > std::numeric_limits<std::size_t>::max() / scratch_slots) {
        throw std::overflow_error("Kernel workspace exceeds the platform limit");
    }
    impl_->scratch_elements = impl_->count * scratch_slots;

    auto target = target_builder(llvm::Triple(llvm::sys::getProcessTriple()), mode);
    auto machine = require(target.createTargetMachine(), "Creating the native target machine");
    auto context = std::make_unique<llvm::LLVMContext>();
    std::vector<std::string> remarks;
    auto module = build_module(graph, impl_->plan, *context, *machine, mode, remarks);
    impl_->info = inspect(*module, *machine, std::move(remarks));
    impl_->jit = require(llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(target))
            .setDataLayout(module->getDataLayout()).create(), "Creating LLJIT");
    impl_->jit->getObjTransformLayer().setTransform(
        [state = impl_.get()](std::unique_ptr<llvm::MemoryBuffer> object)
            -> llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> {
            state->object = object->getBuffer().str();
            return object;
        });
    check(impl_->jit->addIRModule({std::move(module), std::move(context)}), "Adding the kernel IR");
    // Lookup materializes and links the object while its owner is alive.
    impl_->entry = require(impl_->jit->lookup("forge_kernel"), "Looking up the generated kernel")
                       .toPtr<EntryPoint>();
}

CompiledKernel::~CompiledKernel() = default;

const Shape& CompiledKernel::output_shape() const { return impl_->plan.output_shape; }

const std::vector<std::optional<Shape>>& CompiledKernel::input_shapes() const {
    return impl_->plan.input_shapes;
}

const KernelInfo& CompiledKernel::info() const { return impl_->info; }
const std::string& CompiledKernel::object() const { return impl_->object; }

Workspace CompiledKernel::make_workspace() const {
    return Workspace(input_shapes().size(), impl_->plan.slot_count, impl_->scratch_elements);
}

void CompiledKernel::run(std::span<const InputBuffer> inputs, OutputBuffer output,
                         Workspace& workspace) const {
    const auto bytes = impl_->count * sizeof(float);
    if (output.size() != bytes || inputs.size() != input_shapes().size()) {
        throw std::invalid_argument("Kernel buffer sizes or argument count do not match");
    }
    if (workspace.inputs_.size() != inputs.size() ||
        workspace.buffers_.size() != impl_->plan.slot_count ||
        workspace.scratch_.size() != impl_->scratch_elements) {
        throw std::invalid_argument("Workspace does not match the compiled kernel");
    }
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (!input_shapes()[index]) {
            continue;
        }
        if (inputs[index].size() != bytes) {
            throw std::invalid_argument("Kernel input size does not match the compiled shape");
        }
        if (overlaps(inputs[index], output)) {
            throw std::invalid_argument("Kernel output must be disjoint from all inputs");
        }
        workspace.inputs_[index] = inputs[index].data();
    }
    if (impl_->count == 0) {
        return;
    }
    workspace.buffers_[0] = output.data();
    for (std::size_t slot = 1; slot < workspace.buffers_.size(); ++slot) {
        workspace.buffers_[slot] = workspace.scratch_.data() + (slot - 1) * impl_->count;
    }
    impl_->entry(impl_->count, workspace.inputs_.data(), workspace.buffers_.data());
}

Assembly emit_x86_linux(const Graph& graph, std::size_t input_count, KernelMode mode) {
    initialize_targets();
    auto target = target_builder(llvm::Triple("x86_64-unknown-linux-gnu"), mode);
    auto machine = require(target.createTargetMachine(), "Creating the Linux x86-64 target");
    llvm::LLVMContext context;
    std::vector<std::string> remarks;
    auto module = build_module(graph, detail::plan_kernel(graph, input_count), context, *machine,
                               mode, remarks);
    auto info = inspect(*module, *machine, std::move(remarks));
    llvm::SmallVector<char> buffer;
    llvm::raw_svector_ostream stream(buffer);
    llvm::legacy::PassManager emitter;
    if (machine->addPassesToEmitFile(emitter, stream, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
        throw std::runtime_error("LLVM could not emit x86-64 assembly");
    }
    emitter.run(*module);
    return {std::move(info), std::string(stream.str())};
}

} // namespace forge
