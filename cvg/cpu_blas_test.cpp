#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl.h>
#include "cpu_blas_test.h"
#include "util.h"

int cpu_sgemm(int loops, int M, int N, int K, float alpha, float beta)
{
    printf("Intel MKL sgemm: loops=%d M=%d N=%d K=%d alpha=%f beta=%f\n",loops,M,N,K,alpha,beta);

    float *a, *b, *c;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, M, b, K, beta, c, M);
    }
    stop = clock();

    summarize_sgemm(c, loops, M, N, K, alpha, beta, start, stop);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);
    
    return 0;
}

int cpu_dgemm(int loops, int M, int N, int K, double alpha, double beta)
{
    printf("Intel MKL dgemm: loops=%d M=%d N=%d K=%d alpha=%f beta=%f\n", loops, M, N, K, alpha, beta);

    double *a, *b, *c;
    new_double_matrix(a, M, K);
    new_double_matrix(b, K, N);
    new_double_matrix(c, M, N);

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, M, b, K, beta, c, M);
    }
    stop = clock();

    summarize_dgemm(c, loops, M, N, K, alpha, beta, start, stop);

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);

    return 0;
}

int cpu_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta)
{
    printf("Intel MKL ssyrkgemm: loops=%d M=%d N=%d K=%d\n", loops, M, N, K);
    assert(M == N);

    float *a, *b, *c;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);

    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans, N, K, alpha, a, M, beta, c, N);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, M, b, K, beta, c, M);
    }
    stop = clock();

    summarize_ssyrkgemm(c, loops, M, N, K, alpha, beta, start, stop);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);

    return 0;
}

int cpu_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta)
{
    printf("Intel MKL dsyrkgemm: loops=%d M=%d N=%d K=%d\n", loops, M, N, K);
    assert(M == N);

    double *a, *b, *c;
    new_double_matrix(a, M, K);
    new_double_matrix(b, K, N);
    new_double_matrix(c, M, N);
    
    clock_t start, stop;
    start = clock();
    for (int i = 0; i < loops; ++i) {
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, N, K, alpha, a, M, beta, c, N);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, M, b, K, beta, c, M);
    }
    stop = clock();

    summarize_dsyrkgemm(c, loops, M, N, K, alpha, beta, start, stop);

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);

    return 0;
}


