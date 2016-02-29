
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <windows.h>
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

int main_gpu_test(int loops, int M, int N, int K)
{
    printf("CUBLAS sgemm: loops=%d M=%d N=%d K=%d\n", loops, M, N, K);
    
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

    printf("sgemm_multiply(). Elapsed time = %g seconds\n",
        ((double)(stop - start)) / CLOCKS_PER_SEC);

    printf("C:\n");
    pr_array(c, N);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cublasDestroy(handle);

    return 0;

}
