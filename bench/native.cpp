#include "graph.h"
#include "jit.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

enum class Workload { add, demo, chain };

void measure(benchmark::State& state, Workload workload) {
    const auto size = static_cast<std::size_t>(state.range(0));
    const auto mode = state.range(1) == 0 ? forge::KernelMode::scalar : forge::KernelMode::avx2;
    const auto fusion = state.range(2) == 0 ? forge::FusionMode::disabled : forge::FusionMode::enabled;
    if (mode == forge::KernelMode::avx2 && !forge::host_has_avx2()) {
        state.SkipWithMessage("AVX2 is unavailable on this host");
        return;
    }

    forge::GraphBuilder builder;
    const auto input_node = builder.input(0, {size}, {});
    const auto two = builder.literal(2.0F, {});
    auto result = builder.binary(forge::Op::add, input_node, two, {});
    if (workload != Workload::add) {
        const auto three = builder.literal(3.0F, {});
        result = builder.binary(forge::Op::multiply, result, three, {});
    }
    if (workload == Workload::demo) {
        result = builder.sqrt(result, {});
    }
    if (workload == Workload::chain) {
        const auto four = builder.literal(4.0F, {});
        const auto five = builder.literal(5.0F, {});
        result = builder.binary(forge::Op::add, result, four, {});
        result = builder.binary(forge::Op::multiply, result, five, {});
    }

    const forge::CompiledKernel kernel(std::move(builder).finish(result), 1, mode, fusion);
    auto workspace = kernel.make_workspace();
    std::vector<float> input(size);
    std::vector<float> output(size);
    for (std::size_t i = 0; i < size; ++i) {
        input[i] = static_cast<float>(i % 81);
    }
    const std::array inputs{forge::InputBuffer(std::as_bytes(std::span(input)))};
    const auto destination = std::as_writable_bytes(std::span(output));

    for (auto _ : state) {
        kernel.run(inputs, destination, workspace);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));
    state.counters["loops"] = static_cast<double>(kernel.plan().loop_count());
    state.counters["temporary_arrays"] = static_cast<double>(kernel.plan().temporary_arrays());
    // These three workloads read one input and write one output per materialized loop.
    state.counters["modeled_bytes_per_element"] =
        static_cast<double>(kernel.plan().loop_count() * 2 * sizeof(float));
    state.SetLabel(kernel.info().features);
}

void add(benchmark::State& state) { measure(state, Workload::add); }
void demo(benchmark::State& state) { measure(state, Workload::demo); }
void chain(benchmark::State& state) { measure(state, Workload::chain); }

void cases(benchmark::Benchmark* benchmark_case) {
    benchmark_case->ArgNames({"elements", "avx2", "fusion"});
    for (const auto size : {9, 1024, 65536, 1048576}) {
        for (const auto avx2 : {0, 1}) {
            for (const auto fusion : {0, 1}) {
                benchmark_case->Args({size, avx2, fusion});
            }
        }
    }
}

// A wall clock avoids coarse per-process CPU timer ticks on Windows.
BENCHMARK(add)->Apply(cases)->UseRealTime();
BENCHMARK(demo)->Apply(cases)->UseRealTime();
BENCHMARK(chain)->Apply(cases)->UseRealTime();

} // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::AddCustomContext("comparison", "Forge fused vs unfused on matched targets, preallocated, wall clock");
    benchmark::AddCustomContext("host_has_avx2", forge::host_has_avx2() ? "yes" : "no");
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
