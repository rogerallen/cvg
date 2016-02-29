/*
* cvg - simple cpu vs. gpu sgemm perf test
*/
#include <stdio.h>
#include <stdlib.h>
#include "gpu_blas_test.h"
#include "cpu_blas_test.h"

int main(int argc, char **argv) {
    int ic = 1; // index argc
    int rc;

    char sel = 'g';
    int loops = 1;
    int m = 1024, n = 1024, k = 2048;
    int block = 1024;

    if (ic < argc) sel = argv[ic++][0];
    switch (sel) {
    case 'g':
        if (ic < argc) loops = atoi(argv[ic++]);
        if (ic < argc) m = atoi(argv[ic++]);
        if (ic < argc) n = atoi(argv[ic++]);
        if (ic < argc) k = atoi(argv[ic++]);
        rc = cublas_gpu_test(loops, m, n, k);
        break;
    case 'x':
        if (ic < argc) loops = atoi(argv[ic++]);
        if (ic < argc) m = atoi(argv[ic++]);
        if (ic < argc) n = atoi(argv[ic++]);
        if (ic < argc) k = atoi(argv[ic++]);
        if (ic < argc) block = atoi(argv[ic++]);
        rc = cublasxt_gpu_test(loops, m, n, k, block);
        break;
    case 'c':
        if (ic < argc) loops = atoi(argv[ic++]);
        if (ic < argc) m = atoi(argv[ic++]);
        if (ic < argc) n = atoi(argv[ic++]);
        if (ic < argc) k = atoi(argv[ic++]);
        rc = main_cpu_test(loops, m, n, k);
        break;
    default:
        printf("unknown select %c\n",sel);
        rc = 1;
        break;
    }
    return rc;
}