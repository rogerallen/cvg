#ifndef GPU_BLAS_TEST_H
#define GPU_BLAS_TEST_H
int cublas_gpu_test(int loops, int M, int N, int K);
int cublasxt_gpu_test(int loops, int M, int N, int K, int block_size);
#endif