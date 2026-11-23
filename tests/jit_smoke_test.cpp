#include <gtest/gtest.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <memory>
#include <string>
#include <utility>

TEST(JitSmokeTest, CompilesAndExecutesNativeFunction) {
    ASSERT_FALSE(llvm::InitializeNativeTarget());
    ASSERT_FALSE(llvm::InitializeNativeTargetAsmPrinter());

    auto jit = llvm::orc::LLJITBuilder().create();
    ASSERT_TRUE(static_cast<bool>(jit)) << llvm::toString(jit.takeError());

    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("forge_smoke", *context);
    // LLJIT uses the current process ABI, including its data layout.
    module->setTargetTriple((*jit)->getTargetTriple());
    module->setDataLayout((*jit)->getDataLayout());

    const auto features = llvm::sys::getHostCPUFeatures();
    llvm::outs() << "JIT target: " << module->getTargetTriple().str()
                 << "\nData layout: " << module->getDataLayoutStr()
                 << "\nHost CPU: " << llvm::sys::getHostCPUName()
                 << ", AVX: " << (features.lookup("avx") ? "yes" : "no")
                 << ", AVX2: " << (features.lookup("avx2") ? "yes" : "no") << '\n';

    auto *float_type = llvm::Type::getFloatTy(*context);
    auto *add_type = llvm::FunctionType::get(float_type, {float_type}, false);
    auto *add = llvm::Function::Create(add_type, llvm::Function::ExternalLinkage,
                                       "forge_smoke_add_two", *module);
    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(*context, "entry", add));
    auto *two = llvm::ConstantFP::get(float_type, 2.0);
    builder.CreateRet(builder.CreateFAdd(add->getArg(0), two));

    std::string verification_error;
    llvm::raw_string_ostream verification_log(verification_error);
    ASSERT_FALSE(llvm::verifyModule(*module, &verification_log)) << verification_error;

    if (auto error = (*jit)->addIRModule({std::move(module), std::move(context)})) {
        FAIL() << llvm::toString(std::move(error));
    }

    // Lookup materializes and links the function. Keep the JIT alive for both calls.
    auto symbol = (*jit)->lookup("forge_smoke_add_two");
    ASSERT_TRUE(static_cast<bool>(symbol)) << llvm::toString(symbol.takeError());
    auto *add_two = symbol->toPtr<float(float)>();

    EXPECT_EQ(add_two(40.5f), 42.5f);
    EXPECT_EQ(add_two(-5.0f), -3.0f);
}
