#ifndef UTIL_H
#define UTIL_H
void new_float_matrix(float* &x, long height, long width);
void delete_float_matrix(float* &x);
void pr_array(float *c, int ld);
void summarize(float *c, int loops, int M, int N, int K, clock_t start, clock_t stop);
#endif