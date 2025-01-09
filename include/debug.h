#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

extern inline void _print_vector(const float* vec, int size, const char* desr){
    if (desr){
        printf("\n%s \n", desr);
    }
    for(int i=0;i<size;i++){
        printf("%.2f ", vec[i]);
    }
    printf("\n");
}
#define printVector(vec, size, desr) _print_vector(vec, size, desr)

extern inline void _printMatrix(float* mat, int row, int col, const char* desc){
    printf("\n %s\n", desc);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            printf("%.2f ", mat[i*col+j]);
        }
        printf("\n");
    }
}

#define printMatrix(mat, row, col, desc) _printMatrix(mat, row, col, desc)


#endif