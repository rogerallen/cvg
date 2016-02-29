#include "stdio.h"
#include "util.h"

// FIXME column & row major matrices
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void new_float_matrix(float* &x, long height, long width)
{
    x = new float[height * width];
    for (long i = 0; i < height; ++i) {
        for (long j = 0; j < width; ++j) {
            x[i * width + j] = ((i*j) % 10) / 10.0f;
        }
    }
}

void delete_float_matrix(float* &x) 
{
    delete[] x;
}

void pr_array(float *x, int width)
{
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%7.5f ", x[i * width + j]);
        }
        printf("...[snip]...\n");
    }
    printf("...[snip]...\n");
}