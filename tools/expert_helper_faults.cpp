#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Test-only process interposition. Never linked into or committed with the engine.
static std::atomic<bool> fired{false}, runtime_started{false};
static std::atomic<int> allocations{0};
static thread_local cudaError_t synthetic_launch_error = cudaSuccess;
static bool mode(const char* name) {
    const char* value = std::getenv("HELPER_FAULT_TEST");
    return value && std::strcmp(value, name) == 0;
}
static bool helper_device() {
    using Get = cudaError_t (*)(int*);
    static auto get = reinterpret_cast<Get>(dlsym(RTLD_NEXT, "cudaGetDevice"));
    int device = -1;
    return get && get(&device) == cudaSuccess && device == 1;
}
static cudaError_t fail(const char* operation) {
    std::fprintf(stderr, "FAULT_TEST injected %s on helper GPU\n", operation);
    return cudaErrorInvalidValue;
}
extern "C" cudaError_t cudaMalloc(void** pointer, size_t bytes) {
    using Fn = cudaError_t (*)(void**, size_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaMalloc"));
    if (mode("allocation") && helper_device() && ++allocations == 3 && !fired.exchange(true))
        return fail("allocation"); // Exercise cleanup after the pool and an input buffer exist.
    return real(pointer, bytes);
}
extern "C" cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes,
                                       cudaMemcpyKind kind, cudaStream_t stream) {
    using Fn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
    if (helper_device() && kind == cudaMemcpyHostToDevice) {
        if (bytes > 1024 * 1024 && mode("prepare") && !fired.exchange(true)) return fail("prepare");
        if (bytes > 1024 && bytes < 128 * 1024) {
            runtime_started = true;
            if (mode("input") && !fired.exchange(true)) return fail("input");
        }
    }
    return real(dst, src, bytes, kind, stream);
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaStream_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    const auto status = real(stream); // Retire real GPU work before simulating a completion error.
    if (status == cudaSuccess && helper_device() && runtime_started && mode("completion") && !fired.exchange(true))
        return fail("completion");
    return status;
}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* function, void** args) {
    using Fn = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaLaunchKernelExC"));
    if (mode("kernel") && helper_device() && !fired.exchange(true)) {
        synthetic_launch_error = cudaErrorInvalidConfiguration;
        return fail("kernel");
    }
    return real(config, function, args);
}
#if CUDART_VERSION >= 13000
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
                                         void** args, size_t shared, cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaKernel_t, dim3, dim3, void**, size_t, cudaStream_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    if (mode("kernel") && helper_device() && !fired.exchange(true)) {
        synthetic_launch_error = cudaErrorInvalidConfiguration;
        return fail("kernel");
    }
    return real(kernel, grid, block, args, shared, stream);
}
#endif
extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                       void** args, size_t shared, cudaStream_t stream) {
    using Fn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    if (mode("kernel") && helper_device() && !fired.exchange(true)) {
        synthetic_launch_error = cudaErrorInvalidConfiguration;
        return fail("kernel");
    }
    return real(function, grid, block, args, shared, stream);
}
extern "C" cudaError_t cudaGetLastError() {
    using Fn = cudaError_t (*)();
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaGetLastError"));
    auto status = real();
    if (synthetic_launch_error != cudaSuccess) {
        status = synthetic_launch_error;
        synthetic_launch_error = cudaSuccess;
    }
    return status;
}
