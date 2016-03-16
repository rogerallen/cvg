#ifndef CPU_BLAS_TEST_H
#define CPU_BLAS_TEST_H
int cpu_sgemm(int loops, int M, int N, int K, float alpha, float beta);
int cpu_dgemm(int loops, int M, int N, int K, double alpha, double beta);
int cpu_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta);
int cpu_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta);
#endif