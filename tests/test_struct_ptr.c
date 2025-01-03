#include<stdio.h>
#include<stdlib.h>

typedef struct {
    int* x;
    int* y;
    int* z;
    int size;
} B;

typedef struct {
    B b;
    int* mem;
}A;

int main(int argc, char const *argv[])
{
    A a;
    a.mem = (int*)malloc(6*sizeof(int));
    for(int i=0;i<6; i++){
        a.mem[i] = i;
    }
    a.b.x = a.mem;
    a.b.size = 2;
    a.b.y = a.mem + 2;
    a.b.z = a.mem + 4;

    B* b = &(a.b);
    for(int i=0;i<2; i++){
        printf("i:%d bx: %d\n",i, b->x[i]);
        printf("i:%d by: %d\n",i, b->y[i]);
        printf("i:%d bz: %d\n",i, b->z[i]);
        b->x[i] += 1;
        b->y[i] += 2;
        b->z[i] += 4;
    }
    b->size = 6;

    for (int i = 0; i < 6; i++){
        printf("i: %d, mem: %d \n", i, a.mem[i]);
    }
    printf("size: %d\n", a.b.size);

    free(a.mem);
    return 0;
}

