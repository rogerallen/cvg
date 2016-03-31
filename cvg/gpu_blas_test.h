#ifndef GPU_BLAS_TEST_H
#define GPU_BLAS_TEST_H
int gpu_cublas_sgemm(int loops, int M, int N, int K, float alpha, float beta, int num_gpus, int *gpu_ids, bool csv_output);
int gpu_cublas_dgemm(int loops, int M, int N, int K, double alpha, double beta, int num_gpus, int *gpu_ids, bool csv_output);
int gpu_cublasxt_sgemm(int loops, int M, int N, int K, float alpha, float beta, int block_size, int num_gpus, int *gpu_ids, bool csv_output);
int gpu_cublasxt_dgemm(int loops, int M, int N, int K, double alpha, double beta, int block_size, int num_gpus, int *gpu_ids, bool csv_output);
int gpu_cublas_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta, int num_gpus, int *gpu_ids, bool csv_output);
int gpu_cublas_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta, int num_gpus, int *gpu_ids, bool csv_output);
int gpu_cublasxt_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta, int block_size, int num_gpus, int *gpu_ids, bool csv_output);
int gpu_cublasxt_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta, int block_size, int num_gpus, int *gpu_ids, bool csv_output);
#endif
