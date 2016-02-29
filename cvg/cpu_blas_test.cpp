#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl.h>
#include "cpu_blas_test.h"
#include "util.h"

int main_cpu_test(int loops, int M, int N, int K)
{
    printf("Intel MKL sgemm: loops=%d M=%d N=%d K=%d\n",loops,M,N,K);

    float *a, *b, *c;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);

    clock_t start, stop;
    start = clock();
    float alpha = 1.11f, beta = 0.91f;
    for (int i = 0; i < loops; ++i) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, M, b, K, beta, c, M);
    }
    stop = clock();

    summarize(c, loops, M, N, K, start, stop);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);
    
    return 0;
}



