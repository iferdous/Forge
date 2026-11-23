#include "graph.h"
#include "jit.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace forge {
namespace {

enum class GuardPosition { before, after };

class GuardedBuffer {
public:
    explicit GuardedBuffer(std::size_t bytes, GuardPosition position = GuardPosition::after)
        : bytes_(bytes) {
#ifdef _WIN32
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        const std::size_t page = info.dwPageSize;
#else
        const auto page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) {
            throw std::runtime_error("Could not query the memory page size");
        }
        const auto page = static_cast<std::size_t>(page_size);
#endif
        const auto writable = std::max(page, ((bytes + page - 1) / page) * page);
        allocation_ = writable + page;
        const auto guard_offset = position == GuardPosition::before ? 0 : writable;
#ifdef _WIN32
        base_ = static_cast<std::byte*>(
            VirtualAlloc(nullptr, allocation_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!base_) {
            throw std::runtime_error("Could not allocate guarded test memory");
        }
        DWORD old_permissions;
        if (!VirtualProtect(base_ + guard_offset, page, PAGE_NOACCESS, &old_permissions)) {
            VirtualFree(base_, 0, MEM_RELEASE);
            throw std::runtime_error("Could not protect the test guard page");
        }
#else
        void* memory = mmap(nullptr, allocation_, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED) {
            throw std::runtime_error("Could not allocate guarded test memory");
        }
        base_ = static_cast<std::byte*>(memory);
        if (mprotect(base_ + guard_offset, page, PROT_NONE) != 0) {
            munmap(base_, allocation_);
            throw std::runtime_error("Could not protect the test guard page");
        }
#endif
        data_ = base_ + (position == GuardPosition::before ? page : writable - bytes_);
    }

    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;

    ~GuardedBuffer() {
#ifdef _WIN32
        VirtualFree(base_, 0, MEM_RELEASE);
#else
        munmap(base_, allocation_);
#endif
    }

    OutputBuffer data() { return {data_, bytes_}; }

private:
    std::byte* base_;
    std::byte* data_;
    std::size_t bytes_;
    std::size_t allocation_;
};

Graph binary_graph(std::size_t size, bool chain = false) {
    GraphBuilder builder;
    auto left = builder.input(0, {size}, {});
    auto right = builder.input(1, {size}, {});
    auto result = builder.binary(Op::add, left, right, {});
    if (chain) {
        const auto half = builder.literal(0.5F, {});
        result = builder.binary(Op::multiply, result, half, {});
    }
    return std::move(builder).finish(result);
}

TEST(Avx2Jit, GuardsBothInputAndOutputAtTheScalarRemainder) {
    if (!host_has_avx2()) {
        GTEST_SKIP() << "Native AVX2 is unavailable";
    }
    for (const auto size : {0U, 1U, 7U, 8U, 9U, 15U, 16U, 17U, 1023U, 1024U, 1025U}) {
        SCOPED_TRACE(size);
        const auto bytes = size * sizeof(float);
        GuardedBuffer lhs(bytes), rhs(bytes), destination(bytes);
        std::vector<float> left(size), right(size), expected(size);
        for (std::size_t i = 0; i < size; ++i) {
            left[i] = static_cast<float>(i % 71);
            right[i] = static_cast<float>(i % 13) * 0.25F;
            expected[i] = (left[i] + right[i]) * 0.5F;
        }
        if (bytes != 0) {
            std::memcpy(lhs.data().data(), left.data(), bytes);
            std::memcpy(rhs.data().data(), right.data(), bytes);
        }
        const std::array inputs{InputBuffer(lhs.data()), InputBuffer(rhs.data())};
        const auto graph = binary_graph(size, true);
        for (auto fusion : {FusionMode::disabled, FusionMode::enabled}) {
            const CompiledKernel kernel(graph, 2, KernelMode::avx2, fusion);
            auto workspace = kernel.make_workspace();
            if (bytes != 0) {
                std::memset(destination.data().data(), 0xA5, bytes);
            }
            kernel.run(inputs, destination.data(), workspace);
            if (bytes != 0) {
                EXPECT_EQ(std::memcmp(destination.data().data(), expected.data(), bytes), 0);
            }
        }
    }
}

TEST(Avx2Jit, ChecksHostFeaturesBeforeRunningAvx2) {
    if (host_has_avx2()) {
        const CompiledKernel kernel(binary_graph(9), 2, KernelMode::avx2);
        EXPECT_NE(kernel.info().features.find("+avx2"), std::string::npos);
        EXPECT_NE(kernel.info().ir.find("<8 x float>"), std::string::npos);
        EXPECT_FALSE(kernel.info().remarks.empty());
    } else {
        EXPECT_THROW(CompiledKernel(binary_graph(9), 2, KernelMode::avx2), std::invalid_argument);
    }

    const auto emitted = emit_x86_linux(binary_graph(9), 2, KernelMode::avx2);
    EXPECT_NE(emitted.text.find("vaddps"), std::string::npos);
    EXPECT_NE(emitted.text.find("vaddss"), std::string::npos);
}

TEST(GeneratedCodeBounds, GuardsOrderedReductionsAtBothEnds) {
    for (const auto size : {1U, 7U, 8U, 9U, 31U, 33U}) {
        for (const auto op : {Op::reduce_sum, Op::reduce_max, Op::reduce_mean}) {
            GraphBuilder builder;
            const auto input = builder.input(0, {size}, {});
            const auto result = builder.reduce(op, input, {});
            const auto graph = std::move(builder).finish(result);
            for (const auto mode : {KernelMode::scalar, KernelMode::avx2}) {
                if (mode == KernelMode::avx2 && !host_has_avx2()) {
                    continue;
                }
                const CompiledKernel kernel(graph, 1, mode);
                for (const auto side : {GuardPosition::before, GuardPosition::after}) {
                    SCOPED_TRACE(::testing::Message() << op_name(op) << " length " << size
                                                     << " mode " << static_cast<int>(mode)
                                                     << " guard " << static_cast<int>(side));
                    std::vector<float> ones(size, 1.0F);
                    GuardedBuffer source(size * sizeof(float), side), dest(sizeof(float), side);
                    std::memcpy(source.data().data(), ones.data(), source.data().size());
                    const std::array inputs{InputBuffer(source.data())};
                    auto workspace = kernel.make_workspace();
                    kernel.run(inputs, dest.data(), workspace);
                    float actual = 0;
                    std::memcpy(&actual, dest.data().data(), sizeof(float));
                    EXPECT_EQ(actual, op == Op::reduce_sum ? static_cast<float>(size) : 1.0F);
                }
            }
        }
    }
}

TEST(GeneratedCodeBounds, GuardsOddMatmulTilesAndEmptyInnerDimension) {
    constexpr std::size_t m = 17, n = 35;
    for (const std::size_t k : {0, 1, 33}) {
        GraphBuilder builder;
        const auto a = builder.input(0, {m, k}, {});
        const auto b = builder.input(1, {k, n}, {});
        const auto product = builder.matmul(a, b, {});
        const auto graph = std::move(builder).finish(product);
        for (const auto mode : {KernelMode::scalar, KernelMode::avx2}) {
            if (mode == KernelMode::avx2 && !host_has_avx2()) {
                continue;
            }
            const CompiledKernel kernel(graph, 2, mode);
            for (const auto side : {GuardPosition::before, GuardPosition::after}) {
                SCOPED_TRACE(::testing::Message() << "inner dimension " << k
                                                 << " mode " << static_cast<int>(mode)
                                                 << " guard " << static_cast<int>(side));
                GuardedBuffer left(m * k * sizeof(float), side);
                GuardedBuffer right(k * n * sizeof(float), side);
                GuardedBuffer dest(m * n * sizeof(float), side);
                if (k != 0) {
                    std::vector<float> lhs(m * k, 1.0F), rhs(k * n, 2.0F);
                    std::memcpy(left.data().data(), lhs.data(), left.data().size());
                    std::memcpy(right.data().data(), rhs.data(), right.data().size());
                }
                std::memset(dest.data().data(), 0xA5, dest.data().size());
                const std::array inputs{InputBuffer(left.data()), InputBuffer(right.data())};
                auto workspace = kernel.make_workspace();
                kernel.run(inputs, dest.data(), workspace);
                const std::vector<float> expected(m * n, static_cast<float>(k) * 2.0F);
                EXPECT_EQ(std::memcmp(dest.data().data(), expected.data(), dest.data().size()), 0);
            }
        }
    }
}

} // namespace
} // namespace forge
