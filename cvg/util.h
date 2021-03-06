#ifndef UTIL_H
#define UTIL_H
void new_float_matrix(float* &x, long height, long width);
void new_double_matrix(double* &x, long height, long width);
void delete_float_matrix(float* &x);
void delete_double_matrix(double* &x);
void pr_array(float *c, int ld);
void pr_array(double *c, int ld);
void summarize_sgemm(float *c, int loops, int M, int N, int K, float alpha, float beta, clock_t start, clock_t stop, bool csv_output);
void summarize_dgemm(double *c, int loops, int M, int N, int K, double alpha, double beta, clock_t start, clock_t stop, bool csv_output);
void summarize_ssyrkgemm(float *c, int loops, int M, int N, int K, float alpha, float beta, clock_t start, clock_t stop, bool csv_output);
void summarize_dsyrkgemm(double *c, int loops, int M, int N, int K, double alpha, double beta, clock_t start, clock_t stop, bool csv_output);
#endif
