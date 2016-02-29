#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl.h>
#include "cpu_blas_test.h"
#include "util.h"

int main_cpu_test(int loops, int M, int N, int K)
{
    printf("Intel sgemm: loops=%d M=%d N=%d K=%d\n",loops,M,N,K);

    float *a, *b, *c;
    new_float_matrix(a, M, K);
    new_float_matrix(b, K, N);
    new_float_matrix(c, M, N);

    clock_t start, stop;
    start = clock();
    float alpha = 1.11f, beta = 0.91f;
    for (int i = 0; i < loops; ++i) {
        // CPU BLAS is Row Major, GPU is Column
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, K, b, N, beta, c, N);
    }
    stop = clock();

    printf("sgemm_multiply(). Elapsed time = %g seconds\n",
        ((double)(stop - start)) / CLOCKS_PER_SEC);
    
    printf("C:\n");
    pr_array(c, N);

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);
    
    return 0;
}


//DGEMM way. The PREFERED way, especially for large matrices
void Dgemm_multiply(double* a, double*  b, double*  c, int N)
{

}

//initialize array with random data
void init_arr(int N, double* a)
{
    int i, j;
    for (i = 0; i< N; i++) {
        for (j = 0; j<N; j++) {
            a[i*N + j] = (i + j + 1) % 10; //keep all entries less than 10. pleasing to the eye!
        }
    }
}

//print array to std out
void print_arr(int N, char * name, double* array)
{
    int i, j;
    printf("\n%s\n", name);
    for (i = 0; i<N; i++){
        for (j = 0; j<N; j++) {
            printf("%g\t", array[N*i + j]);
        }
        printf("\n");
    }
}



