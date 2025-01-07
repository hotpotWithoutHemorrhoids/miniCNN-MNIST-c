#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

extern inline void _print_vector(float* vec, int size){
    for(int i=0;i<size;i++){
        printf("%.1f ", vec[i]);
    }
    printf("\n");
}
#define printVector(vec, size) _print_vector(vec, size)

extern inline void _printMatrix(float* mat, int row, int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            printf("%.1f ", mat[i*col+j]);
        }
        printf("\n");
    }
}

#define printMatrix(mat, row, col) _printMatrix(mat, row, col)


#endif