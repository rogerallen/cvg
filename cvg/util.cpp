#include <stdio.h>
#include <time.h>
#include "util.h"

// column major matrices
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void new_float_matrix(float* &x, long height, long width)
{
    x = new float[height * width];
    for (long j = 0; j < width; ++j) {
        for (long i = 0; i < height; ++i) {
            x[IDX2C(i, j, height)] = ((i*j) % 10) / 10.0f;
        }
    }
}

void new_double_matrix(double* &x, long height, long width)
{
    x = new double[height * width];
    for (long j = 0; j < width; ++j) {
        for (long i = 0; i < height; ++i) {
            x[IDX2C(i, j, height)] = ((i*j) % 10) / 10.0;
        }
    }
}

void delete_float_matrix(float* &x)
{
    delete[] x;
}

void delete_double_matrix(double* &x)
{
    delete[] x;
}

void pr_array(float *x, int ld)
{
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            printf("%7.5f ", x[IDX2C(i, j, ld)]);
        }
        printf("...[snip]...\n");
    }
    printf("...[snip]...\n");
}

void pr_array(double *x, int ld)
{
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            printf("%7.5f ", x[IDX2C(i, j, ld)]);
        }
        printf("...[snip]...\n");
    }
    printf("...[snip]...\n");
}

void summarize_sgemm(float *c, int loops, int M, int N, int K, clock_t start, clock_t stop)
{
    printf("Result:\n");
    pr_array(c, M);

    double data_bytes = (double)(M*K + K*N + M*N) * sizeof(float);
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("SGEMM: [%dx%d] * [%dx%d] + [%dx%d]\n", M, K, K, N, M, N);
    printf("seconds:     %f\n", timer_seconds);
    printf("Gigabytes:   %.1f\n", data_bytes / 1e9);
    // the total number of floating point operations for a typical *GEMM call 
    // is approximately 2MNK.
    printf("Gigaflops:   %.1f\n", 2.0*M*N*K*loops / 1e9);
    printf("Gigaflops/s: %.1f\n", 2.0*M*N*K*loops / timer_seconds / 1e9);
}

void summarize_dgemm(double *c, int loops, int M, int N, int K, clock_t start, clock_t stop)
{
    printf("Result:\n");
    pr_array(c, M);

    double data_bytes = (double)(M*K + K*N + M*N) * sizeof(float);
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("DGEMM: [%dx%d] * [%dx%d] + [%dx%d]\n", M, K, K, N, M, N);
    printf("seconds:     %f\n", timer_seconds);
    printf("Gigabytes:   %.1f\n", data_bytes / 1e9);
    // the total number of floating point operations for a typical *GEMM call 
    // is approximately 2MNK.
    printf("Gigaflops:   %.1f\n", 2.0*M*N*K*loops / 1e9);
    printf("Gigaflops/s: %.1f\n", 2.0*M*N*K*loops / timer_seconds / 1e9);
}

void summarize_ssyrkgemm(float *c, int loops, int M, int N, int K, clock_t start, clock_t stop)
{
    printf("Result:\n");
    pr_array(c, M);

    double data_bytes = (double)(M*K + K*N + M*N) * sizeof(float);
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    //FIXMEprintf("DSYRK: [%dx%d] * [%dx%d] * %.2f + [%dx%d] * %.2f\n", M, K, K, M, alpha, M, M, beta);
    printf("FIXME SSYRKGEMM: [%dx%d] * [%dx%d] + [%dx%d]\n", M, K, K, N, M, N);
    printf("seconds:     %f\n", timer_seconds);
    printf("Gigabytes:   %.1f\n", data_bytes / 1e9);
    // the total number of floating point operations for a typical *GEMM call 
    // is approximately 2MNK.
    printf("Gigaflops:   %.1f\n", 2.0*M*N*K*loops / 1e9);
    printf("Gigaflops/s: %.1f\n", 2.0*M*N*K*loops / timer_seconds / 1e9);
}

void summarize_dsyrkgemm(double *c, int loops, int M, int N, int K, clock_t start, clock_t stop)
{
    printf("Result:\n");
    pr_array(c, M);

    double data_bytes = (double)(M*K + K*N + M*N) * sizeof(float);
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    //FIXMEprintf("DSYRK: [%dx%d] * [%dx%d] * %.2f + [%dx%d] * %.2f\n", M, K, K, M, alpha, M, M, beta);
    printf("FIXME DSYRKGEMM: [%dx%d] * [%dx%d] + [%dx%d]\n", M, K, K, N, M, N);
    printf("seconds:     %f\n", timer_seconds);
    printf("Gigabytes:   %.1f\n", data_bytes / 1e9);
    // the total number of floating point operations for a typical *GEMM call 
    // is approximately 2MNK.
    printf("Gigaflops:   %.1f\n", 2.0*M*N*K*loops / 1e9);
    printf("Gigaflops/s: %.1f\n", 2.0*M*N*K*loops / timer_seconds / 1e9);
}