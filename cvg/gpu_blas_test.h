#ifndef GPU_BLAS_TEST_H
#define GPU_BLAS_TEST_H
int gpu_cublas_sgemm(int loops, int M, int N, int K, float alpha, float beta);
int gpu_cublas_dgemm(int loops, int M, int N, int K, double alpha, double beta);
int gpu_cublasxt_sgemm(int loops, int M, int N, int K, float alpha, float beta, int block_size);
int gpu_cublasxt_dgemm(int loops, int M, int N, int K, double alpha, double beta, int block_size);
int gpu_cublas_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta);
int gpu_cublas_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta);
int gpu_cublasxt_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta, int block_size);
int gpu_cublasxt_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta, int block_size);
#endif