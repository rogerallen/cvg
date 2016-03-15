/*
* cvg - simple cpu vs. gpu sgemm perf test
*/
#include <iostream>
#include <string>
#include "gpu_blas_test.h"
#include "cpu_blas_test.h"

using namespace std;

void print_usage() 
{
    cout << "cvg - simple CPU vs. GPU perf test" << endl;
    cout << "usage:" << endl;
    cout << " cvg <options>" << endl;
    cout << "options:" << endl;
    cout << " -s c|g|x : test select CPU (Intel MKL), GPU (CUBLAS), GPU (CUBLAS-XT)" << endl;
    cout << " -p s|d   : test precision single (default) or double" << endl;
    cout << " -l #     : loops of the algorithm (default 1)" << endl;
    cout << " -m #     : matrix M dimension (default 1024)" << endl;
    cout << " -n #     : matrix N dimension (default 1024)" << endl;
    cout << " -k #     : matrix K dimension (default 2048)" << endl;
    cout << " -b #     : CUBLAS-XT block_dim (default 1024)" << endl;
}

int main(int argc, char **argv) {
    int ic = 1; // index argc
    int rc;

    char sel = 'g';
    char prec = 's';
    int loops = 1;
    int m = 1024, n = 1024, k = 2048;
    int block = 1024;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
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
    switch (sel) {
    case 'g':
        rc = cublas_gpu_test(loops, m, n, k);
        break;
    case 'x':
        rc = cublasxt_gpu_test(loops, m, n, k, block);
        break;
    case 'c':
        rc = main_cpu_test(loops, m, n, k);
        break;
    default:
        cerr << "unknown select: " << sel << endl;
        rc = 1;
        break;
    }
    return rc;
}