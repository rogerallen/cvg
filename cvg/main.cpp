/*
* cvg - simple cpu vs. gpu sgemm perf test
*/
#include <iostream>
#include <string>
#include "gpu_blas_test.h"
#ifndef NO_CPU_BLAS
#include "cpu_blas_test.h"
#endif

using namespace std;

void print_usage()
{
    cout << "cvg - simple CPU vs. GPU perf test" << endl;
#ifdef NO_CPU_BLAS
    cout << "!!! NO CPU BLAS (Intel MKL) INCLUDED IN THIS EXE !!!" << endl;
#endif
    cout << "usage:" << endl;
    cout << " cvg <options>" << endl;
    cout << "options:" << endl;
    cout << " -t g|s   : test type: GEMM, SYRK+GEMM (default GEMM)" << endl;
    cout << " -s c|g|x : API select: CPU (Intel MKL), GPU (CUBLAS), GPU (CUBLAS-XT)" << endl;
    cout << " -p s|d   : test precision: single (default) or double" << endl;
    cout << " -l #     : loops of the algorithm (default 1)" << endl;
    cout << " -m #     : matrix M dimension (default 1024)" << endl;
    cout << " -n #     : matrix N dimension (default 1024)" << endl;
    cout << " -k #     : matrix K dimension (default 2048)" << endl;
    cout << " -b #     : CUBLAS-XT block_dim (default 1024)" << endl;
    cout << " -A #     : alpha GEMM parameter (default 1.11)" << endl;
    cout << " -B #     : beta GEMM parameter (default 0.91)" << endl;
    cout << " -g #     : [CUBLAS-XT only] gpu id(s) to use. can specify multiple -g options. (default=0) " << endl;
    cout << " -c       : compact CSV output" << endl;
}

int main(int argc, char **argv) {
    int rc;

    char test = 'g';
    char sel = 'g';
    char prec = 's';
    int loops = 1;
    int m = 1024, n = 1024, k = 2048;
    int block = 1024;
    double alpha = 1.11, beta = 0.91;
    int num_gpus = 0;
    int gpu_ids[4] = { 0, 1, 2, 3 };
    bool csv_output = false;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
            case 't':
                test = argv[++i][0];
                break;
            case 's':
                sel = argv[++i][0];
                break;
            case 'p':
                prec = argv[++i][0];
                break;
            case 'l':
                loops = stoi(argv[++i]);
                break;
            case 'm':
                m = stoi(argv[++i]);
                break;
            case 'n':
                n = stoi(argv[++i]);
                break;
            case 'k':
                k = stoi(argv[++i]);
                break;
            case 'b':
                block = stoi(argv[++i]);
                break;
            case 'A':
                alpha = stod(argv[++i]);
                break;
            case 'B':
                beta = stod(argv[++i]);
                break;
            case 'g':
                gpu_ids[num_gpus++] = stoi(argv[++i]);
                break;
            case 'c':
                csv_output = true;
                break;
            default:
                cerr << "ERROR: unknown option -" << argv[i][1] << endl;
                print_usage();
                exit(1);
            }
        } else {
            cerr << "ERROR: unknown argument" << argv[i] << endl;
            print_usage();
            return(1);
        }
    }

    if (num_gpus == 0) {
        num_gpus = 1;
    }

#ifndef NO_CPU_BLAS
    if ((sel == 'c') && (test == 'g') && (prec == 's')) {
        rc = cpu_sgemm(loops, m, n, k, (float)alpha, (float)beta, csv_output);
    } else if ((sel == 'c') && (test == 'g') && (prec == 'd')) {
        rc = cpu_dgemm(loops, m, n, k, alpha, beta, csv_output);
    } else
#endif
    if ((sel == 'g') && (test == 'g') && (prec == 's')) {
        rc = gpu_cublas_sgemm(loops, m, n, k, (float)alpha, (float)beta, csv_output);
    } else if ((sel == 'g') && (test == 'g') && (prec == 'd')) {
        rc = gpu_cublas_dgemm(loops, m, n, k, alpha, beta, csv_output);
    } else if ((sel == 'x') && (test == 'g') && (prec == 's')) {
        rc = gpu_cublasxt_sgemm(loops, m, n, k, (float)alpha, (float)beta, block, num_gpus, gpu_ids, csv_output);
    } else if ((sel == 'x') && (test == 'g') && (prec == 'd')) {
        rc = gpu_cublasxt_dgemm(loops, m, n, k, alpha, beta, block, num_gpus, gpu_ids, csv_output);
#ifndef NO_CPU_BLAS
    } else if ((sel == 'c') && (test == 's') && (prec == 's')) {
        rc = cpu_ssyrkgemm(loops, m, n, k, (float)alpha, (float)beta, csv_output);
    } else if ((sel == 'c') && (test == 's') && (prec == 'd')) {
        rc = cpu_dsyrkgemm(loops, m, n, k, alpha, beta, csv_output);
#endif
    } else if ((sel == 'g') && (test == 's') && (prec == 's')) {
        rc = gpu_cublas_ssyrkgemm(loops, m, n, k, (float)alpha, (float)beta, csv_output);
    } else if ((sel == 'g') && (test == 's') && (prec == 'd')) {
        rc = gpu_cublas_dsyrkgemm(loops, m, n, k, alpha, beta, csv_output);
    } else if ((sel == 'x') && (test == 's') && (prec == 's')) {
        rc = gpu_cublasxt_ssyrkgemm(loops, m, n, k, (float)alpha, (float)beta, block, num_gpus, gpu_ids, csv_output);
    } else if ((sel == 'x') && (test == 's') && (prec == 'd')) {
        rc = gpu_cublasxt_dsyrkgemm(loops, m, n, k, alpha, beta, block, num_gpus, gpu_ids, csv_output);
    } else {
        cerr << "ERROR: no test selected." << endl;
        rc = 3;
    }
    return rc;
}
