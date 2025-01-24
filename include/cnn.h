#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include "dataloader.h"

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

#define INPUT_SIZE 784

#define ImageSize 28
#define K1 5
#define C1 16
#define P1 2
#define K2 5
#define C2 36
#define P2 2
#define FC1_SIZE 128
#define OUTPUT_SIZE 10

#define EPOCHS 10
#define BATCH 1000
#define TRAIN_SPLIT 0.8
#define LEARN_RATE 0.01f
#define MOMENTUM 0.9f
#define EPSILON 0.000000001f

typedef struct{
    int x;
    int y;
    int z;
} Shape;

void printShape(const Shape *s, const char* desc){
    printf("%s shape x:%d, y: %d, z: %d\n", desc, s->x, s->y,s->z);
}

typedef struct
{
    Shape in_size;
    Shape out_size;
}Size;
void init_size(Size *size, int in_x,int in_y,int in_z,int out_x,int out_y,int out_z);

typedef struct{
    float* weights;
    Size size;
    int kernel_size;
    int stride;
    int filters;
    int weights_size;
    int num_params;
} Conv;

void init_conv(Conv* conv, int in_x,int in_y, int in_channel, int kernel_size, int stride, int filters);

typedef struct
{
    Size size;
    int pool_size;
}Pool;

void init_pool(Pool* pool, int in_x, int in_y, int in_z, int pool_size);

typedef struct{
    float* weights;
    float* bias;
    Size size;
    int weight_size;
    int bias_size;
    int num_params;
}FC;

void init_fc(FC* fc, int in_size, int out_size);

typedef struct{
    Conv conv1; // (K1,K1,C1)
    Pool pool1;
    Conv conv2; // (K2,K2,C2)
    Pool pool2;
    // Conv Conv3; // (K3,K3,C3)
    FC fc1;
    FC fc2;
}Paramerters;

typedef struct{
    float* out_conv1; // ( (imageSize-K1)/stride + 1, (imageSize-K1)/stride + 1, C1)
    Shape conv1_size;
    float* out_pool1; // ( ((imageSize-K1)/stride + 1)/P1, ((imageSize-K1)/stride + 1)/P1, C1)
    Shape pool1_size;
    float* out_conv2; // ( ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, C2)
    Shape conv2_size;
    float* out_pool2; // ( (((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1)/P2, (((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1)/P2, C2)
    Shape pool2_size;
    float* out_fc1; 
    Shape fc_size;
    float* out_fc2; // (1,1,10)
    Shape output_size;
}Activation;

typedef struct{
    float* grad_out_conv1;
    float* grad_out_pool1;
    float* grad_out_conv2;
    float* grad_out_pool2;
    float* grad_out_fc1;
    float* grad_out_fc2;
}Grad_Activation;

typedef struct{
    float* mem_fc2;
    float* mem_fc1;
    float* mem_conv2;
    float* mem_conv1;
}Mementun;

typedef struct
{
    Data datas;
    int imageSize;
    Paramerters params;
    float* params_memory;
    int toal_params;
    Activation acts;
    float* acts_memory;
    int total_acts;
    Grad_Activation grad_acts;
    float* grad_acts_memory;
    int total_grad_acts;

    float* mementun_memory;
    Mementun mementun;
}CNN;

void init_params(float*params, int size);
void CNN_init(CNN *model, int imageSize,int k1, int c1, int stride1, int p1,int k2, int c2, int stride2,int p2,int fc1_size, int outputSize, int batch);
void initialize_memory(float* mem, int size);
void CNN_clear(CNN *model);
Shape conv_forward(float *inp, Shape in_shape,float* out, float* conv_weights, int kernel_size, int stride, int channel);
Shape pool_froward(float* inp, int h, int w, int z, float* out, int pool_size);
Shape fc_forward(float* inp, int inp_size, float* out, float* weights, int output_size, float* bias, bool acti_func);
void softmax_forward(float* inp, int inp_size);
void cnn_forward(CNN *model, float* inp, int h, int w);

void softmax_backward(float* inp, int inp_size,int target, float* d_inp);
void fc_backward(float* inp, Shape inp_size, float* d_loss, Shape out_size, float* weights, float* bias, float* d_inp, float* mementun, float lr, bool acti_func);
void pool_backward(float* inp, Shape inp_size, float* d_loss, Shape out_size, float* d_inp, int pool_size);
void conv_backward(float* inp, Shape inp_size, float*d_loss, Shape out_size, float* out, 
                    float* d_inp, float* conv_weights, float* mementun, int kernel_size, 
                    int stride, int channel,float lr);

void cnn_backward(CNN *model,float* inp,int label, float lr, int output_size);

#endif