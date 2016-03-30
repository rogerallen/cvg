#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl.h>
#include <windows.h>
#pragma comment(lib, "user32.lib")
#include "cpu_blas_test.h"
#include "util.h"

void list_cpu_info()
{
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo);
    printf("Hardware information: \n");
    printf("  OEM ID:                %u\n", siSysInfo.dwOemId);
    printf("  Number of processors:  %u\n", siSysInfo.dwNumberOfProcessors);
    HKEY hKey;
    DWORD dwMHz, BufSize;
    long lError = RegOpenKeyEx(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0,
        KEY_READ,
        &hKey);
    if (lError == ERROR_SUCCESS) {
        RegQueryValueEx(hKey, "~MHz", NULL, NULL, (LPBYTE)&dwMHz, &BufSize);
        printf("  CPU clock:             %d MHz\n", dwMHz);
    }
}

int cpu_sgemm(int loops, int M, int N, int K, float alpha, float beta, bool csv_output)
{
    if (!csv_output) {
        printf("Intel MKL sgemm: loops=%d M=%d N=%d K=%d alpha=%f beta=%f\n", loops, M, N, K, alpha, beta);
        list_cpu_info();
    } else {
        printf("Intel MKL sgemm,%d,%d,%d,%d,%f,%f", loops, M, N, K, alpha, beta);
    }

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

    summarize_sgemm(c, loops, M, N, K, alpha, beta, start, stop, csv_output);
    if (csv_output) {
        printf("\n");
    }

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);

    return 0;
}

int cpu_dgemm(int loops, int M, int N, int K, double alpha, double beta, bool csv_output)
{
    if (!csv_output) {
        printf("Intel MKL dgemm: loops=%d M=%d N=%d K=%d alpha=%f beta=%f\n", loops, M, N, K, alpha, beta);
        list_cpu_info();
    } else {
        printf("Intel MKL dgemm,%d,%d,%d,%d,%f,%f", loops, M, N, K, alpha, beta);
    }

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

    summarize_dgemm(c, loops, M, N, K, alpha, beta, start, stop, csv_output);
    if (csv_output) {
        printf("\n");
    }

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);

    return 0;
}

int cpu_ssyrkgemm(int loops, int M, int N, int K, float alpha, float beta, bool csv_output)
{
    if (!csv_output) {
        printf("Intel MKL ssyrkgemm: loops=%d M=%d N=%d K=%d alpha=%f beta=%f\n", loops, M, N, K, alpha, beta);
        list_cpu_info();
    } else {
        printf("Intel MKL ssyrkgemm,%d,%d,%d,%d,%f,%f", loops, M, N, K, alpha, beta);
    }
    
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

    summarize_ssyrkgemm(c, loops, M, N, K, alpha, beta, start, stop, csv_output);
    if (csv_output) {
        printf("\n");
    }

    delete_float_matrix(a);
    delete_float_matrix(b);
    delete_float_matrix(c);

    return 0;
}

int cpu_dsyrkgemm(int loops, int M, int N, int K, double alpha, double beta, bool csv_output)
{
    if (!csv_output) {
        printf("Intel MKL dsyrkgemm: loops=%d M=%d N=%d K=%d alpha=%f beta=%f\n", loops, M, N, K, alpha, beta);
        list_cpu_info();
    }
    else {
        printf("Intel MKL dsyrkgemm,%d,%d,%d,%d,%f,%f", loops, M, N, K, alpha, beta);
    }

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

    summarize_dsyrkgemm(c, loops, M, N, K, alpha, beta, start, stop, csv_output);
    if (csv_output) {
        printf("\n");
    }

    delete_double_matrix(a);
    delete_double_matrix(b);
    delete_double_matrix(c);

    return 0;
}
