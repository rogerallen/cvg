#include <time.h>
#include <stdio.h>
#include <windows.h>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "gpu_blas_test.h"
#include "util.h"

#define HANDLE_CUDA_ERROR( err ) ( HandleCudaError( err, __FILE__, __LINE__ ) )
static void HandleCudaError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_CUBLAS_ERROR( err, str ) ( HandleCublasError( err, __FILE__, __LINE__, str) )
static void HandleCublasError(cublasStatus_t err, const char *file, int line, const char *str)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        printf("error %s %d in %s at line %d\n", str, err, // FIXME why no error code?
            file, line);
        exit(EXIT_FAILURE);
    }
}

void list_cuda_devices() 
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
    }
}

int main_gpu_test(int loops, int M, int N, int K)
{
    printf("NVIDIA CUBLAS sgemm: loops=%d M=%d N=%d K=%d\n", loops, M, N, K);
    
    list_cuda_devices();

    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&handle),"cublasCreate fail");

    float *a, *b, *c;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);

    // time all the extra stuff for setting up the matrices
    clock_t start, stop;
    start = clock();
    float *dev_a, *dev_b, *dev_c;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_a, M*K*sizeof(*a)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_b, K*N*sizeof(*b)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_c, M*N*sizeof(*c)));
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, K, sizeof(*a), a, M, dev_a, M), "cublasSetMatrix A fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(K, N, sizeof(*b), b, K, dev_b, K), "cublasSetMatrix B fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, N, sizeof(*c), c, M, dev_c, M), "cublasSetMatrix C fail");

    float alpha = 1.11f, beta = 0.91f;
    for (int i = 0; i < loops; ++i) {
        // FIXME this is column-major, CPU is row-major
        HANDLE_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_a, M, dev_b, K, &beta, dev_c, M), "Sgemm fail");
    }
    HANDLE_CUBLAS_ERROR(cublasGetMatrix(M, N, sizeof(*c), dev_c, M, c, M), "cublasGetMatrix C fail");
    stop = clock();

    printf("Result:\n");
    pr_array(c, M);

    double data_bytes = (double)(M*K + K*N + M*N) * sizeof(float);
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("SGEMM: [%dx%d] * [%dx%d] + [%dx%d]\n", M, K, K, N, M, N);
    printf("seconds:     %f\n", timer_seconds);
    printf("Gigabytes:   %.1f\n", data_bytes / 1e9);
    // the total number of floating point operations for a typical *GEMM call 
    // is approximately 2MNK.
    printf("Gigaflops:   %.1f\n", 2.0*M*N*K*loops / 1e9);
    printf("Gigaflops/s: %.1f\n", 2.0*M*N*K*loops / timer_seconds / 1e9);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cublasDestroy(handle);

    return 0;

}
