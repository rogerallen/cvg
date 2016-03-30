#ifndef CPU_BLAS_TEST_H
#define CPU_BLAS_TEST_H
int cpu_sgemm(int loops, int M, int N, int K, float alpha, float beta, bool csv_output);
int cpu_dgemm(int loops, int M, int N, int K, double alpha, double beta, bool csv_output);
int cpu_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta, bool csv_output);
int cpu_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta, bool csv_output);
#endif
