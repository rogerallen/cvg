#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
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
        printf("error %s %d in %s at line %d\n", str, err, // why no cublasGetErrorString?
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

int gpu_cublas_sgemm(int loops, int M, int N, int K, float alpha, float beta)
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

    for (int i = 0; i < loops; ++i) {
        HANDLE_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_a, M, dev_b, K, &beta, dev_c, M), "Sgemm fail");
    }
    HANDLE_CUBLAS_ERROR(cublasGetMatrix(M, N, sizeof(*c), dev_c, M, c, M), "cublasGetMatrix C fail");
    stop = clock();

    summarize_sgemm(c, loops, M, N, K, start, stop);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cublasDestroy(handle);

    return 0;

}

int gpu_cublas_dgemm(int loops, int M, int N, int K, double alpha, double beta)
{
    printf("NVIDIA CUBLAS dgemm: loops=%d M=%d N=%d K=%d\n", loops, M, N, K);

    list_cuda_devices();

    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&handle), "cublasCreate fail");

    double *a, *b, *c;
    new_double_matrix(a, M, K);
    new_double_matrix(b, K, N);
    new_double_matrix(c, M, N);

    // time all the extra stuff for setting up the matrices
    clock_t start, stop;
    start = clock();
    double *dev_a, *dev_b, *dev_c;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_a, M*K*sizeof(*a)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_b, K*N*sizeof(*b)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_c, M*N*sizeof(*c)));
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, K, sizeof(*a), a, M, dev_a, M), "cublasSetMatrix A fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(K, N, sizeof(*b), b, K, dev_b, K), "cublasSetMatrix B fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, N, sizeof(*c), c, M, dev_c, M), "cublasSetMatrix C fail");

    for (int i = 0; i < loops; ++i) {
        HANDLE_CUBLAS_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_a, M, dev_b, K, &beta, dev_c, M), "Sgemm fail");
    }
    HANDLE_CUBLAS_ERROR(cublasGetMatrix(M, N, sizeof(*c), dev_c, M, c, M), "cublasGetMatrix C fail");
    stop = clock();

    summarize_dgemm(c, loops, M, N, K, start, stop);

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cublasDestroy(handle);

    return 0;

}

int gpu_cublasxt_sgemm(int loops, int M, int N, int K, float alpha, float beta, int block_dim, int num_gpus, int *gpu_ids)
{
    printf("NVIDIA CUBLASXT sgemm: loops=%d M=%d N=%d K=%d block_dim=%d\n", loops, M, N, K, block_dim);

    list_cuda_devices();

    cublasXtHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasXtCreate(&handle), "cublasXtCreate fail");
    
    HANDLE_CUBLAS_ERROR(cublasXtDeviceSelect(handle, num_gpus, gpu_ids), "cublasXtDeviceSelect fail");

    HANDLE_CUBLAS_ERROR(cublasXtSetBlockDim(handle, block_dim), "cublasXtSetBlockDim fail");
    
    float *a, *b, *c;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        HANDLE_CUBLAS_ERROR(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a, M, b, K, &beta, c, M), "Sgemm fail");
    }
    stop = clock();

    summarize_sgemm(c, loops, M, N, K, start, stop);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);

    cublasXtDestroy(handle);

    return 0;

}

int gpu_cublasxt_dgemm(int loops, int M, int N, int K, double alpha, double beta, int block_dim, int num_gpus, int *gpu_ids)
{
    printf("NVIDIA CUBLASXT dgemm: loops=%d M=%d N=%d K=%d block_dim=%d\n", loops, M, N, K, block_dim);

    list_cuda_devices();

    cublasXtHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasXtCreate(&handle), "cublasXtCreate fail");

    HANDLE_CUBLAS_ERROR(cublasXtDeviceSelect(handle, num_gpus, gpu_ids), "cublasXtDeviceSelect fail");

    HANDLE_CUBLAS_ERROR(cublasXtSetBlockDim(handle, block_dim), "cublasXtSetBlockDim fail");

    double *a, *b, *c;
    new_double_matrix(a, M, K);
    new_double_matrix(b, K, N);
    new_double_matrix(c, M, N);

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        HANDLE_CUBLAS_ERROR(cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a, M, b, K, &beta, c, M), "Sgemm fail");
    }
    stop = clock();

    summarize_dgemm(c, loops, M, N, K, start, stop);

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);

    cublasXtDestroy(handle);

    return 0;

}

int gpu_cublas_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta)
{
    printf("NVIDIA CUBLAS sgemm: loops=%d M=%d N=%d K=%d\n", loops, M, N, K);
    assert(K > N);

    list_cuda_devices();

    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&handle), "cublasCreate fail");

    float *a, *b, *c, *d;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);
    new_float_matrix(d, K, K);

    // time all the extra stuff for setting up the matrices
    clock_t start, stop;
    start = clock();
    float *dev_a, *dev_b, *dev_c, *dev_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_a, M*K*sizeof(*a)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_b, K*N*sizeof(*b)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_c, M*N*sizeof(*c)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_d, K*K*sizeof(*d)));
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, K, sizeof(*a), a, M, dev_a, M), "cublasSetMatrix A fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(K, N, sizeof(*b), b, K, dev_b, K), "cublasSetMatrix B fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, N, sizeof(*c), c, M, dev_c, M), "cublasSetMatrix C fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(K, K, sizeof(*d), d, K, dev_d, K), "cublasSetMatrix D fail");

    for (int i = 0; i < loops; ++i) {
        HANDLE_CUBLAS_ERROR(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, M, &alpha, a, K, &beta, d, K), "Ssyrk fail");
        HANDLE_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_a, M, dev_b, K, &beta, dev_c, M), "Sgemm fail");
    }
    HANDLE_CUBLAS_ERROR(cublasGetMatrix(M, N, sizeof(*c), dev_c, M, c, M), "cublasGetMatrix C fail");
    stop = clock();

    summarize_sgemm(c, loops, M, N, K, start, stop);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);
    delete_float_matrix(d);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    cublasDestroy(handle);

    return 0;

}

int gpu_cublas_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta)
{
    printf("NVIDIA CUBLAS dgemm: loops=%d M=%d N=%d K=%d\n", loops, M, N, K);
    assert(K > N);

    list_cuda_devices();

    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&handle), "cublasCreate fail");

    double *a, *b, *c, *d;
    new_double_matrix(a, M, K);
    new_double_matrix(b, K, N);
    new_double_matrix(c, M, N);
    new_double_matrix(d, K, K);

    // time all the extra stuff for setting up the matrices
    clock_t start, stop;
    start = clock();
    double *dev_a, *dev_b, *dev_c, *dev_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_a, M*K*sizeof(*a)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_b, K*N*sizeof(*b)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_c, M*N*sizeof(*c)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_d, K*K*sizeof(*d)));
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, K, sizeof(*a), a, M, dev_a, M), "cublasSetMatrix A fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(K, N, sizeof(*b), b, K, dev_b, K), "cublasSetMatrix B fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(M, N, sizeof(*c), c, M, dev_c, M), "cublasSetMatrix C fail");
    HANDLE_CUBLAS_ERROR(cublasSetMatrix(K, K, sizeof(*d), d, K, dev_d, K), "cublasSetMatrix D fail");

    for (int i = 0; i < loops; ++i) {
		HANDLE_CUBLAS_ERROR(cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, M, &alpha, a, K, &beta, d, K), "Dsyrk fail");
        HANDLE_CUBLAS_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_a, M, dev_b, K, &beta, dev_c, M), "Dgemm fail");
    }
    HANDLE_CUBLAS_ERROR(cublasGetMatrix(M, N, sizeof(*c), dev_c, M, c, M), "cublasGetMatrix C fail");
    stop = clock();

    summarize_dgemm(c, loops, M, N, K, start, stop);

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);
    delete_double_matrix(d);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    cublasDestroy(handle);

    return 0;

}

int gpu_cublasxt_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta, int block_dim, int num_gpus, int *gpu_ids)
{
    printf("NVIDIA CUBLASXT sgemm: loops=%d M=%d N=%d K=%d block_dim=%d\n", loops, M, N, K, block_dim);
    assert(K > N);

    list_cuda_devices();

    cublasXtHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasXtCreate(&handle), "cublasXtCreate fail");

    HANDLE_CUBLAS_ERROR(cublasXtDeviceSelect(handle, num_gpus, gpu_ids), "cublasXtDeviceSelect fail");

    HANDLE_CUBLAS_ERROR(cublasXtSetBlockDim(handle, block_dim), "cublasXtSetBlockDim fail");

    float *a, *b, *c, *d;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);
    new_float_matrix(d, K, K);

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        HANDLE_CUBLAS_ERROR(cublasXtSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, N, &alpha, a, K, &beta, d, K), "Ssyrk fail");
        HANDLE_CUBLAS_ERROR(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a, M, b, K, &beta, c, M), "Sgemm fail");
    }
    stop = clock();

    summarize_sgemm(c, loops, M, N, K, start, stop);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);
    delete_float_matrix(d);

    cublasXtDestroy(handle);

    return 0;

}

int gpu_cublasxt_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta, int block_dim, int num_gpus, int *gpu_ids)
{
    printf("NVIDIA CUBLASXT dgemm: loops=%d M=%d N=%d K=%d block_dim=%d\n", loops, M, N, K, block_dim);
    assert(K > N);

    list_cuda_devices();

    cublasXtHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasXtCreate(&handle), "cublasXtCreate fail");

    HANDLE_CUBLAS_ERROR(cublasXtDeviceSelect(handle, num_gpus, gpu_ids), "cublasXtDeviceSelect fail");

    HANDLE_CUBLAS_ERROR(cublasXtSetBlockDim(handle, block_dim), "cublasXtSetBlockDim fail");

    double *a, *b, *c, *d;
    new_double_matrix(a, M, K);
    new_double_matrix(b, K, N);
    new_double_matrix(c, M, N);
    new_double_matrix(d, K, K);

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        HANDLE_CUBLAS_ERROR(cublasXtDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, N, &alpha, a, K, &beta, d, K), "Dsyrk fail");
        HANDLE_CUBLAS_ERROR(cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a, M, b, K, &beta, c, M), "Dgemm fail");
    }
    stop = clock();

    summarize_dgemm(c, loops, M, N, K, start, stop);

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);
    delete_double_matrix(d);

    cublasXtDestroy(handle);

    return 0;

}