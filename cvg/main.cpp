// Add: View->Property Pages->Configuration->Intel

#include <stdio.h>
#include <stdlib.h>
#include "gpu_blas_test.h"
#include "cpu_blas_test.h"

int main(int argc, char **argv) {
    int ic = 1; // index argc
    int rc;

    int sel   = 0;  // GPU=0, CPU=1
    int loops = 1;
    int m     = 10, n = 10, k = 20;
    int block = 2048;

    if (argc > 1) {
        sel = atoi(argv[ic++]);
    }
    switch (sel) {
    case 0:
        if (ic < argc) loops = atoi(argv[ic++]);
        if (ic < argc) m = atoi(argv[ic++]);
        if (ic < argc) n = atoi(argv[ic++]);
        if (ic < argc) k = atoi(argv[ic++]);
        //if (ic < argc) block = atoi(argv[ic++]);
        rc = main_gpu_test(loops, m, n, k);
        break;
    case 1:
        if (ic < argc) loops = atoi(argv[ic++]);
        if (ic < argc) m = atoi(argv[ic++]);
        if (ic < argc) n = atoi(argv[ic++]);
        if (ic < argc) k = atoi(argv[ic++]);
        rc = main_cpu_test(loops, m, n, k);
        break;
    default:
        printf("no test selected\n");
        rc = 1;
        break;
    }
    return rc;
}