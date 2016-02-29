#include "stdio.h"
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

void delete_float_matrix(float* &x) 
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