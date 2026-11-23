#include "graph.h"
#include "jit.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

void measure(benchmark::State& state, bool demo) {
    const auto size = static_cast<std::size_t>(state.range(0));
    const auto mode = state.range(1) == 0 ? forge::KernelMode::scalar : forge::KernelMode::avx2;
    if (mode == forge::KernelMode::avx2 && !forge::host_has_avx2()) {
        state.SkipWithMessage("AVX2 is unavailable on this host");
        return;
    }

    forge::GraphBuilder builder;
    const auto input_node = builder.input(0, {size}, {});
    const auto two = builder.literal(2.0F, {});
    auto result = builder.binary(forge::Op::add, input_node, two, {});
    if (demo) {
        const auto three = builder.literal(3.0F, {});
        result = builder.sqrt(builder.binary(forge::Op::multiply, result, three, {}), {});
    }

    const forge::CompiledKernel kernel(std::move(builder).finish(result), 1, mode);
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
    state.SetLabel(kernel.info().features);
}

void add(benchmark::State& state) { measure(state, false); }
void demo(benchmark::State& state) { measure(state, true); }

void cases(benchmark::Benchmark* benchmark_case) {
    benchmark_case->ArgNames({"elements", "avx2"});
    for (const auto size : {9, 1024, 65536, 1048576}) {
        benchmark_case->Args({size, 0});
        benchmark_case->Args({size, 1});
    }
}

BENCHMARK(add)->Apply(cases);
BENCHMARK(demo)->Apply(cases);

} // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::AddCustomContext("comparison", "Forge scalar vs AVX2, preallocated");
    benchmark::AddCustomContext("host_has_avx2", forge::host_has_avx2() ? "yes" : "no");
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
