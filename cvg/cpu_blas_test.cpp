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

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);
    
    return 0;
}



