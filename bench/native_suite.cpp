#include "graph.h"
#include "jit.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

enum class Workload { e1, e2, e3, e4, e5, e6, r1, r2, r3, m1 };
constexpr std::uint32_t seed = 620261122;

float fixture_value(std::size_t index, std::size_t input, bool nonnegative) {
    // Match the unsigned integer hash in suite.py exactly.
    auto value = static_cast<std::uint32_t>(index) + seed +
                 static_cast<std::uint32_t>(input) * 0x9E3779B9U;
    value = (value ^ (value >> 16)) * 0x7FEB352DU;
    value = (value ^ (value >> 15)) * 0x846CA68BU;
    value ^= value >> 16;
    return static_cast<float>(value >> 16) * 0x1p-16F - (nonnegative ? 0.0F : 0.5F);
}

forge::Graph build_graph(Workload workload, const std::vector<forge::Shape>& shapes) {
    using forge::Op;
    forge::GraphBuilder builder;
    const auto x = builder.input(0, shapes[0], {});
    const auto input = [&](std::size_t index) { return builder.input(index, shapes[index], {}); };
    const auto binary = [&](Op op, forge::NodeId left, forge::NodeId right) {
        return builder.binary(op, left, right, {});
    };
    const auto scalar = [&](Op op, forge::NodeId left, float right) {
        return binary(op, left, builder.literal(right, {}));
    };
    auto root = x;
    switch (workload) {
    case Workload::e1: root = binary(Op::add, x, input(1)); break;
    case Workload::e2: root = scalar(Op::multiply, x, 3); break;
    case Workload::e3: root = builder.sqrt(x, {}); break;
    case Workload::e4:
        root = binary(Op::multiply, x, input(1));
        root = binary(Op::add, root, input(2));
        break;
    case Workload::e5:
        root = scalar(Op::add, x, 2);
        root = scalar(Op::multiply, root, 3);
        root = builder.sqrt(root, {});
        break;
    case Workload::e6:
        root = scalar(Op::add, x, 1);
        root = scalar(Op::multiply, root, 2);
        root = scalar(Op::add, root, 3);
        root = scalar(Op::multiply, root, 4);
        break;
    case Workload::r1: root = builder.reduce(Op::reduce_sum, x, {}); break;
    case Workload::r2: root = builder.reduce(Op::reduce_max, x, {}); break;
    case Workload::r3: root = builder.reduce(Op::reduce_mean, x, {}); break;
    case Workload::m1: root = builder.matmul(x, input(1), {}); break;
    }
    return std::move(builder).finish(root);
}

float rounded_product(float left, float right) {
    const volatile float result = left * right;
    return result;
}

float reference(Workload workload, const std::vector<std::vector<float>>& arrays, std::size_t index,
                std::size_t k, std::size_t n) {
    const auto& x = arrays[0];
    if (workload == Workload::m1) {
        float total = 0;
        for (std::size_t inner = 0; inner < k; ++inner) {
            total += rounded_product(x[(index / n) * k + inner], arrays[1][inner * n + index % n]);
        }
        return total;
    }
    if (workload == Workload::r1 || workload == Workload::r3) {
        float total = 0;
        for (const auto value : x) {
            total += value;
        }
        return workload == Workload::r3 ? total / static_cast<float>(x.size()) : total;
    }
    switch (workload) {
    case Workload::e1: return x[index] + arrays[1][index];
    case Workload::e2: return x[index] * 3.0F;
    case Workload::e3: return std::sqrt(x[index]);
    case Workload::e4: return rounded_product(x[index], arrays[1][index]) + arrays[2][index];
    case Workload::e5: return std::sqrt((x[index] + 2.0F) * 3.0F);
    case Workload::e6: return (rounded_product(x[index] + 1.0F, 2.0F) + 3.0F) * 4.0F;
    case Workload::r2: return *std::max_element(x.begin(), x.end());
    default: throw std::logic_error("Unexpected workload");
    }
}

void measure(benchmark::State& state, Workload workload) {
    const auto m = static_cast<std::size_t>(state.range(0));
    const auto k = static_cast<std::size_t>(state.range(1));
    const auto n = static_cast<std::size_t>(state.range(2));
    const auto mode = state.range(3) ? forge::KernelMode::avx2 : forge::KernelMode::scalar;
    const auto fusion = state.range(4) ? forge::FusionMode::enabled : forge::FusionMode::disabled;
    if (mode == forge::KernelMode::avx2 && !forge::host_has_avx2()) {
        state.SkipWithMessage("AVX2 is unavailable on this host");
        return;
    }

    std::vector<forge::Shape> shapes{{m}};
    if (workload == Workload::m1) {
        shapes = {{m, k}, {k, n}};
    } else if (workload == Workload::e4) {
        shapes = {{m}, {m}, {m}};
    } else if (workload == Workload::e1) {
        shapes.push_back({m});
    }
    const auto graph = build_graph(workload, shapes);
    const forge::CompiledKernel kernel(graph, shapes.size(), mode, fusion);
    auto workspace = kernel.make_workspace();
    std::vector<std::vector<float>> arrays(shapes.size());
    std::vector<forge::InputBuffer> inputs;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        auto& input = arrays[i];
        input.resize(forge::element_count(shapes[i]));
        for (std::size_t j = 0; j < input.size(); ++j) {
            input[j] = fixture_value(j, i, workload == Workload::e3);
        }
        inputs.emplace_back(std::as_bytes(std::span(input)));
    }
    std::vector<float> output(forge::element_count(kernel.output_shape()));
    const auto destination = std::as_writable_bytes(std::span(output));
    kernel.run(inputs, destination, workspace);
    const auto verify = [&](std::size_t index) {
        const auto expected = reference(workload, arrays, index, k, n);
        return std::bit_cast<std::uint32_t>(output[index]) == std::bit_cast<std::uint32_t>(expected);
    };
    if (!verify(0) || (output.size() > 1 && (!verify(output.size() / 2) || !verify(output.size() - 1)))) {
        state.SkipWithError("Native fixture disagrees with its fixed-order reference");
        return;
    }
    for (auto _ : state) {
        kernel.run(inputs, destination, workspace);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.counters["loops"] = static_cast<double>(kernel.plan().loop_count());
    state.counters["temporary_arrays"] = static_cast<double>(kernel.plan().temporary_arrays());
    if (workload <= Workload::e6) {
        std::size_t accesses = 0;
        for (const auto& region : kernel.plan().regions) {
            ++accesses;  // Each loop writes one output.
            for (const auto id : region.arguments) {
                const auto& node = graph.nodes()[id];
                accesses += !node.shape.empty() && node.op != forge::Op::splat;
            }
        }
        state.counters["modeled_bytes_per_element"] = static_cast<double>(accesses * sizeof(float));
    }
    state.SetLabel(kernel.info().features);
}

void register_suite() {
    constexpr std::pair<std::string_view, Workload> workloads[]{
        {"E1", Workload::e1}, {"E2", Workload::e2}, {"E3", Workload::e3},
        {"E4", Workload::e4}, {"E5", Workload::e5}, {"E6", Workload::e6},
        {"R1", Workload::r1}, {"R2", Workload::r2}, {"R3", Workload::r3}, {"M1", Workload::m1},
    };
    for (const auto& [name, workload] : workloads) {
        auto* bench = benchmark::RegisterBenchmark(std::string(name), measure, workload);
        bench->ArgNames({"m", "k", "n", "avx2", "fusion"})->UseRealTime();
        for (const int avx2 : {0, 1}) {
            if (workload == Workload::m1) {
                for (const auto size : {16, 64, 256, 512}) {
                    bench->Args({size, size, size, avx2, 1});
                }
                bench->Args({31, 63, 17, avx2, 1});
                continue;
            }
            const auto add_length = [&](auto size) {
                bench->Args({size, 0, 0, avx2, 1});
                if (workload <= Workload::e6) {
                    bench->Args({size, 0, 0, avx2, 0});
                }
            };
            for (const auto size : {32, 1000, 1024, 65536, 1048576, 8388608}) {
                add_length(size);
            }
            if (workload == Workload::e5) {
                add_length(1009);
                add_length(67108864);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::AddCustomContext("comparison", "Forge native preallocated wall clock");
    benchmark::AddCustomContext("fixture_seed", std::to_string(seed));
    benchmark::AddCustomContext("host_has_avx2", forge::host_has_avx2() ? "yes" : "no");
    register_suite();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
