#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
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
#define LEARN_RATE 0.05
#define MOMENTUM 0.9f

typedef struct{
    int x;
    int y;
    int z;
} Shape;

typedef struct
{
    Shape in_size;
    Shape out_size;
}Size;
void init_size(Size *size, int in_x,int in_y,int in_z,int out_x,int out_y,int out_z){
    size->in_size.x = in_x;
    size->in_size.y = in_y;
    size->in_size.z = in_z;
    size->out_size.x = out_x;
    size->out_size.y = out_y;
    size->out_size.z = out_z;
}

typedef struct{
    float* weights;
    Size size;
    int kernel_size;
    int stride;
    int filters;
    int weights_size;
    int num_params;
} Conv;

void init_conv(Conv* conv, int in_x,int in_y, int in_channel, int kernel_size, int stride, int filters){
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->filters = filters;
    init_size(&(conv->size), in_x, in_y, in_channel, 
                (in_x - kernel_size)/stride + 1, (in_y - kernel_size)/stride + 1, filters);
    conv->weights_size = kernel_size*kernel_size*filters;
    conv->num_params = conv->weights_size;
    conv->weights = NULL;
}

typedef struct
{
    Size size;
    int pool_size;
}Pool;

void init_pool(Pool* pool, int in_x, int in_y, int in_z, int pool_size){
    init_size(&(pool->size), in_x, in_y, in_z, in_x/pool_size, in_y/pool_size, in_z);
    pool->pool_size = pool_size;
}

typedef struct{
    float* weights;
    float* bias;
    Size size;
    int weight_size;
    int bias_size;
    int num_params;
}FC;

void init_fc(FC* fc, int in_size, int out_size){
    init_size(&(fc->size), 1, 1, in_size, 1, 1, out_size);
    fc->weight_size = in_size*out_size;
    fc->bias_size = out_size;
    fc->num_params = fc->weight_size + fc->bias_size;
    fc->weights = NULL;
    fc->bias = NULL;
}

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

void init_params(float*params, int size){
    float scale = sqrt(2.0f/size);
    for(int i = 0; i < size; i++){
        params[i] = 2*scale * ((float)rand()/(float)(RAND_MAX) - 0.5f);
    }
}

void CNN_init(CNN *model, int imageSize,int k1, int c1, int stride1, int p1,int k2, int c2, int stride2,int p2,int fc1_size, int outputSize, int batch){
    printf("init model\n");
    model->imageSize = imageSize;
    init_conv(&(model->params.conv1), imageSize, imageSize, 1, k1, stride1, c1);
    init_pool(&(model->params.pool1), 
                model->params.conv1.size.out_size.x, 
                model->params.conv1.size.out_size.y, 
                model->params.conv1.size.out_size.z,
                p1);
    init_conv(&(model->params.conv2), model->params.pool1.size.out_size.x, 
                model->params.pool1.size.out_size.y, 
                model->params.pool1.size.out_size.z, 
                k2, stride2, c2);
    init_pool(&(model->params.pool2),
                model->params.conv2.size.out_size.x,
                model->params.conv2.size.out_size.y,
                model->params.conv2.size.out_size.z,
                p2);
    init_fc(&(model->params.fc1), 
             model->params.pool2.size.out_size.x*model->params.pool2.size.out_size.y*model->params.pool2.size.out_size.z, 
             fc1_size);
    init_fc(&(model->params.fc2), fc1_size, outputSize);
    
    model->params_memory = NULL;
    model->acts_memory = NULL;
    model->grad_acts_memory = NULL;
    model->mementun_memory = NULL;
    model->datas.data = NULL;
    model->datas.labels = NULL;

    // int total_params = k1*k1*c1 + k2*k2*c2 + model->params.fc.input_size*fc1_size + fc1_size*outputSize;
    // init weights
    if(model->params_memory == NULL){
        int total_params = model->params.conv1.num_params + model->params.conv2.num_params + model->params.fc1.num_params + model->params.fc2.num_params;
        model->toal_params = total_params;
        printf("total params: %d\n", total_params);
        model->params_memory = (float*)malloc(total_params * sizeof(float));

        int offset = 0;
        model->params.conv1.weights = model->params_memory + offset;
        init_params(model->params.conv1.weights, model->params.conv1.num_params);
        offset += model->params.conv1.num_params;

        model->params.conv2.weights = model->params_memory + offset;
        init_params(model->params.conv2.weights, model->params.conv2.num_params);
        offset += model->params.conv2.num_params;

        model->params.fc1.weights = model->params_memory + offset;
        init_params(model->params.fc1.weights, model->params.fc1.weight_size);
        offset += model->params.fc1.weight_size;

        model->params.fc1.bias = model->params_memory + offset;
        init_params(model->params.fc1.bias, model->params.fc1.bias_size);
        offset += model->params.fc1.bias_size;

        model->params.fc2.weights = model->params_memory + offset;
        init_params(model->params.fc2.weights, model->params.fc2.weight_size);
        offset += model->params.fc2.weight_size;

        model->params.fc2.bias = model->params_memory + offset;
        init_params(model->params.fc2.bias, model->params.fc2.bias_size);
        offset += model->params.fc2.bias_size;
    }

    if(model->datas.data == NULL && model->datas.labels == NULL){
        model->datas.data = (float*)malloc(imageSize*imageSize*batch*sizeof(float));
        model->datas.labels = (int*)malloc(batch*sizeof(int));
    }


    
    if(model->acts_memory == NULL && model->grad_acts_memory == NULL){
        // acts save every layer output
        unsigned int total_acts = 0;
        int offset = 0;
        Paramerters params = model->params;

        total_acts += params.conv1.size.out_size.x * params.conv1.size.out_size.y * params.conv1.size.out_size.z;
        total_acts += params.pool1.size.out_size.x * params.pool1.size.out_size.y * params.pool1.size.out_size.z;
        total_acts += params.conv2.size.out_size.x * params.conv2.size.out_size.y * params.conv2.size.out_size.z;
        total_acts += params.pool2.size.out_size.x * params.pool2.size.out_size.y * params.pool2.size.out_size.z;
        total_acts += params.fc1.size.out_size.x * params.fc1.size.out_size.y * params.fc1.size.out_size.z;
        total_acts += params.fc2.size.out_size.x * params.fc2.size.out_size.y * params.fc2.size.out_size.z;
        model->total_acts = total_acts;

        model->acts_memory = (float*)malloc(total_acts*sizeof(float));
        model->grad_acts_memory = (float*)malloc(total_acts*sizeof(float));

        model->acts.out_conv1 = model->acts_memory + offset;
        model->grad_acts.grad_out_conv1 = model->grad_acts_memory + offset;
        offset += params.conv1.size.out_size.x * params.conv1.size.out_size.y * params.conv1.size.out_size.z;
        
        model->acts.out_pool1 = model->acts_memory + offset;
        model->grad_acts.grad_out_pool1 = model->grad_acts_memory + offset;
        offset += params.pool1.size.out_size.x * params.pool1.size.out_size.y * params.pool1.size.out_size.z;

        model->acts.out_conv2 = model->acts_memory + offset;
        model->grad_acts.grad_out_conv2 = model->grad_acts_memory + offset;
        offset += params.conv2.size.out_size.x * params.conv2.size.out_size.y * params.conv2.size.out_size.z;

        model->acts.out_pool2 = model->acts_memory + offset;
        model->grad_acts.grad_out_pool2 = model->grad_acts_memory + offset;
        offset += params.pool2.size.out_size.x * params.pool2.size.out_size.y * params.pool2.size.out_size.z;

        model->acts.out_fc1 = model->acts_memory + offset;
        model->grad_acts.grad_out_fc1 = model->grad_acts_memory + offset;
        offset += params.fc1.size.out_size.x * params.fc1.size.out_size.y * params.fc1.size.out_size.z;

        model->acts.out_fc2 = model->acts_memory + offset;
        model->grad_acts.grad_out_fc2 = model->grad_acts_memory + offset;
    }


    if(model->mementun_memory == NULL){
        int total_mementun = 0;
        int offset = 0;

        total_mementun += model->params.fc2.num_params;
        total_mementun += model->params.fc1.num_params;
        total_mementun += model->params.conv2.num_params;
        total_mementun += model->params.conv1.num_params;

        model->mementun_memory = (float*)malloc(total_mementun*sizeof(float));

        model->mementun.mem_conv1 = model->mementun_memory + offset;
        offset += model->params.conv1.num_params;

        model->mementun.mem_conv2 = model->mementun_memory + offset;
        offset += model->params.conv2.num_params;

        model->mementun.mem_fc1 = model->mementun_memory + offset;
        offset += model->params.fc1.num_params;

        model->mementun.mem_fc2 = model->mementun_memory + offset;

    }
}

void CNN_clear(CNN *model){
    if(model->params_memory != NULL){
        free(model->params_memory);
        model->params_memory = NULL;
    }
    if(model->acts_memory != NULL){
        free(model->acts_memory);
        model->acts_memory = NULL;
    }
    if(model->grad_acts_memory != NULL){
        free(model->grad_acts_memory);
        model->grad_acts_memory = NULL;
    }
    if(model->mementun_memory != NULL){
        free(model->mementun_memory);
        model->mementun_memory = NULL;
    }
    if (model->datas.data != NULL){
        free(model->datas.data);
        model->datas.data = NULL;
    }
    if (model->datas.labels != NULL){
        free(model->datas.labels);
        model->datas.labels = NULL;
    }
}

Shape conv_forward(float *inp, int h, int w,int z,float* out, float* conv_weights, int kernel_size, int stride, int channel){
    /* 
    inp: (h,w,z)
    out: ((h-kernel_size)/stride+1,(w-kernel_size)/stride+1, channel)
    conv_weights: (kernel_size,kernel_size,channel)
    */
    int out_h = (h-kernel_size)/stride + 1;
    int out_w = (w-kernel_size)/stride + 1;
    Shape output_shape = {out_h, out_w, channel};
    int out_size = out_h*out_w;
    // #pragma omp parallel for
    for (int c = 0; c < channel; c++){
        for (int i = 0; i < out_h; i++){
            for (int j = 0; j < out_w; j++){
                // 输出的第i,j位置应该是输入的i*stride:i*stride+kernel_size,j*stride:j*stride+kernel_size的卷积
                float sum = 0.0f;
                for (int k = 0; k < kernel_size; k++){
                    for (int l = 0; l < kernel_size; l++){
                        sum += inp[(i*stride+k)*w + j*stride+l]*conv_weights[(k*kernel_size+l)*channel + c];
                    }
                }
                // relu
                out[c*out_size + i*out_w + j] = sum>0?sum:0.0f;
            }
        }
    }
    return output_shape;
}

Shape pool_froward(float* inp, int h, int w, int z, float* out, int pool_size){
    int out_h = h/pool_size;
    int out_w = w/pool_size;
    Shape output_shape = {out_h, out_w, z};
    int out_size = out_h*out_w;
    #pragma omp parallel for
    for (int c = 0; c < z; c++){
        float* inp_c = inp + c*h*w;
        // output (i,j) is map input pool (i*pool_size:i*pool_size+pool_size,j*pool_size:j*pool_size+pool_size)
        for (int i = 0; i < out_h; i++){
            for (int j = 0; j < out_w; j++){
                float max = 0.0f;
                for (int k = 0; k < pool_size; k++){
                    for (int l = 0; l < pool_size; l++){
                        max = max > (inp_c[(i*pool_size+k)*w + j*pool_size+l])?max:inp_c[(i*pool_size+k)*w + j*pool_size+l];
                    }
                }
                out[c*out_size + i*out_w + j] = max;
            }
        }
    }
    return output_shape;
}


Shape fc_forward(float* inp, int inp_size, float* out, float* weights, int output_size, float* bias){
    Shape output_shape = {1,1,output_size};

    for(int i = 0; i<output_size; i++){
        out[i] = bias[i];
    }

    for(int i=0;i<inp_size; i++){
        for(int j=0;j<output_size;j++){
            out[j] += inp[i]*weights[i*output_size+j];
        }
    }

    return output_shape;
}

void softmax_forward(float* inp, int inp_size){
    float sum = 0.0f;
    float max = FLT_MIN;
    for(int i=0;i<inp_size;i++){
        max = max>inp[i]?max:inp[i];
    }
    for(int i=0;i<inp_size;i++){
        inp[i] = exp(inp[i]-max);
        sum += inp[i];
    }
    float inv_sum = sum!=0.0f? 1.0f/sum :1.0f;
    for (int i = 0; i < inp_size; i++){
        inp[i] *= inv_sum;
    }
    return;
}
   
void cnn_forward(CNN *model, float* inp, int h, int w){
    Activation* acts = &(model->acts);
    Paramerters* params = &(model->params);
    conv_forward(inp,h,w,1, 
                acts->out_conv1, params->conv1.weights, params->conv1.kernel_size, params->conv1.stride, params->conv1.filters);
    pool_froward(acts->out_conv1,
                params->conv1.size.out_size.y,params->conv1.size.out_size.x, params->conv1.size.out_size.z,
                acts->out_pool1, params->pool1.pool_size);
    conv_forward(acts->out_pool1, params->pool1.size.out_size.y,params->pool1.size.out_size.x, params->pool1.size.out_size.z,
                acts->out_conv2, params->conv2.weights, params->conv2.kernel_size,params->conv2.stride, params->conv2.filters);
    pool_froward(acts->out_conv2, params->conv2.size.out_size.y, params->conv2.size.out_size.x,params->conv2.size.out_size.z,
                acts->out_pool2,params->pool2.pool_size);
    // printf("pool2_shape: %d, %d, %d\n", pool2_shape.x, pool2_shape.y, pool2_shape.z);
    // int flatten_size = pool2_shape.x*pool2_shape.y*pool2_shape.z;
    fc_forward(acts->out_pool2, params->pool2.size.out_size.x*params->pool2.size.out_size.y*params->pool2.size.out_size.z,
                acts->out_fc1, params->fc1.weights, params->fc1.size.out_size.z,params->fc1.bias);
    // fc_forward(acts->fc, fc_shape.z, acts->output, model->params.output.weights, model->params.output.output_size);
    fc_forward(acts->out_fc1, params->fc1.size.out_size.z, 
                acts->out_fc2, params->fc2.weights, params->fc2.size.out_size.z,params->fc2.bias);
    softmax_forward(acts->out_fc2, params->fc2.size.out_size.z);
}

void softmax_backward(float* inp, int inp_size,int target, float* d_inp){
    for(int i=0;i<inp_size;i++){
        if (i == target){
            d_inp[i] = inp[i] - 1.0f;
        }else{
            d_inp[i] = inp[i];
        }
    }
}

void fc_backward(float* inp, Shape inp_size, float* d_loss, Shape out_size, float* weights, float* bias, float* d_inp, float* mementun, float lr){
    /* 
        weights: (inp_len, out_len) 
    */
   //TODO  检查修改
   int inp_len = inp_size.x*inp_size.y*inp_size.z;
   int out_len = out_size.z; // fc 输出1维
    for(int i=0;i<inp_len; i++){
        for(int j=0;j<out_size.z;j++){
            d_inp[i] += d_loss[j]*weights[i*out_size.z+j];
        }
    }
    float* weight_mementun = mementun;
    float* bias_mementun = mementun + inp_len*out_len;

    // update weights
    for (int i = 0; i < inp_len ; i++){
        float* weight_row = weights + i*out_size.z;
        float* mementun_row = weight_mementun + i*out_size.z;
        for (int j = 0; j < out_len; j++){
            float gradW_ij = inp[i]*d_loss[j];
            mementun_row[j] = mementun_row[j]*MOMENTUM - lr*gradW_ij;
            weight_row[j] -= mementun_row[j];
        }
    }

    for(int i=0;i<out_len; i++){
        bias_mementun[i] = bias_mementun[i]*MOMENTUM - lr*d_loss[i];
        bias[i] -= bias_mementun[i];
    }
}

void pool_backward(float* inp, Shape inp_size, float* d_loss, Shape out_size, float* d_inp, int pool_size){
    /* 
        max pool backward
        for a channel z, the loc (i,j) in the output(d_loss)
        map to max value in the inp (i*pool_size:i*pool_size+pool_size, j*pool_size:j*pool_size+pool_size)
    */
   for(int z=0; z<inp_size.z; z++){  // input channel equals output channel
        // init d_inp
        float* d_inp_z = d_inp + z*inp_size.x*inp_size.y;
        float* inp_z = inp + z*inp_size.x*inp_size.y;
        float* d_loss_z = d_loss + z*out_size.x*out_size.y;
        for(int x=0;x<inp_size.x; x++){
            for (int y = 0; y < inp_size.y; y++){
                d_inp_z[x*inp_size.y + y] = 0.0f;
            }
        }
        // update d_inp
        for(int i=0;i<out_size.x;i++){
            for(int j=0;j<out_size.y;j++){
                float max = 0.0f;
                int max_i = 0;
                int max_j = 0;
                for(int k=0;k<pool_size;k++){
                     for(int l=0;l<pool_size;l++){
                        float val = inp_z[(i*pool_size+k)*inp_size.y + j*pool_size+l];
                          if (val > max){
                            max = val;
                            max_i = i*pool_size+k;
                            max_j = j*pool_size+l;
                          }
                     }
                }
                d_inp_z[max_i*inp_size.y + max_j] = d_loss_z[i*out_size.y + j];
            }
        }
   }
}

void conv_backward(float* inp, Shape inp_size, float*d_loss, Shape out_size, float* out, 
                    float* d_inp, float* conv_weights, float* mementun, int kernel_size, 
                    int stride, int channel,float lr){
    /* 
        inp: (h,w,z)
        out: ((h-kernel_size)/stride+1,(w-kernel_size)/stride+1, channel)
        conv_weights: (kernel_size,kernel_size,channel)
        mementun: (kernel_size,kernel_size,channel)
    */
    int out_h = out_size.x;
    int out_w = out_size.y;
    int out_z = out_size.z;
    int inp_h = inp_size.x;
    int inp_w = inp_size.y;
    int inp_z = inp_size.z;
    // relu backward
    for(int z = 0;z<out_z;z++){
        for (int x = 0; x < out_h; x++){
            for (int y = 0; y < out_w; y++){
                d_loss[z*out_h*out_w + x*out_w + y] = out[z*out_h*out_w + x*out_w + y]>0?d_loss[z*out_h*out_w + x*out_w + y]:0.0f;
            }
        }}
    // update weights
    for(int c=0;c<channel;c++){
        for(int i=0;i<kernel_size;i++){
            float* mementun_row = mementun + c*kernel_size*kernel_size + i*kernel_size;
            float* conv_weights_row = conv_weights + c*kernel_size*kernel_size + i*kernel_size;
            for(int j=0;j<kernel_size;j++){
                float grad_w = 0.0f;
                for (int inp_c = 0; inp_c < inp_size.z; inp_c++){
                    float* inp_c_image = inp + inp_c*inp_h*inp_w;
                    for(int l=0;l<out_size.x; l++){
                        for(int k=0;k<out_size.y;k++){
                                grad_w += inp_c_image[(i*stride+l)*out_w+j*stride+k]*d_loss[c*out_h*out_w + l*out_w + k];
                        }
                    }
                mementun_row[j] = mementun_row[j]*MOMENTUM - lr*grad_w;
                // conv_weights_row[j] += mementun_row[j];
            }
        }
        }
    }
    /* 
        for one channel
        suspect input size: (X,Y), kernel_size:K, stride: s=1,
        then output size: ((X-K)+1, (Y-K)+1)
        full model dloss after padding ((X-K)+1+2(K-1), (Y-K)+1+2(K-1)) => (X+K-1, Y+K-1)
        so back conv d_inp_size: (X+K-1-K+1, Y+K-1-K+1) => (X,Y)
    */
    // update d_inp
    if(d_inp != NULL){
        int new_row = out_size.x+2*(kernel_size-1), new_col = out_size.y+2*(kernel_size-1), new_channel = out_size.z; 
        float* full_conv_dloss = (float*)malloc(new_channel*new_row*new_col*sizeof(float));

        for(int inp_c=0;inp_c<inp_size.z; inp_c++){
            float* d_inp_c = d_inp + inp_c*inp_h*inp_w;
            for(int i=0;i<inp_h*inp_w;i++){
                d_inp_c[i] = 0.0f;
            }
        }

        for(int z=0;z<out_z;z++){
            float* full_conv_dloss_z = full_conv_dloss + z*new_row*new_col;
            float* d_loss_z = d_loss + z*out_h*out_w;
            float* conv_weights_z = conv_weights + z*kernel_size*kernel_size;
            // full model padding
            for(int x=0;x<new_row;x++){
                for(int y=0;y<new_col;y++){
                    if (x<kernel_size-1 || x>=out_h+kernel_size-1 || y<kernel_size-1 || y>=out_w+kernel_size-1){
                        full_conv_dloss_z[x*new_col+y] = 0.0f;
                    }else{
                        full_conv_dloss_z[x*new_col+y] = d_loss_z[(x-kernel_size+1)*out_w + y-kernel_size+1];
                    }
                }
            }

            for(int i=0;i<inp_size.x; i++){
                for(int j=0;j<inp_size.y;j++){
                    float d_inp_ij = 0.0f;
                    for (int k = 0; k < kernel_size; k++){
                        for (int l = 0; l < kernel_size; l++){
                            d_inp_ij += full_conv_dloss_z[(i+k)*new_col + j+l]*conv_weights_z[k*kernel_size+l];
                        }
                    }
                    for(int inp_c=0;inp_c<inp_size.z; inp_c++){
                        float* d_inp_c = d_inp + inp_c*inp_h*inp_w;
                        d_inp_c[i*inp_w+j] += d_inp_ij;
                    }
                }
            }
        }
        free(full_conv_dloss);
    }

    for (int i = 0; i < channel*kernel_size*kernel_size; i++){
        conv_weights[i] += mementun[i];
    }
}

void cnn_backward(CNN *model,float* inp,int label, float lr, int output_size){
    Activation* acts = &(model->acts);
    Grad_Activation* grad_acts = &(model->grad_acts);
    Paramerters* params = &(model->params);
    Mementun* mementun = &(model->mementun);
    float* output = acts->out_fc2;
    softmax_backward(output, output_size,label, grad_acts->grad_out_fc2);
    // TODO 更新 fc bias
    fc_backward(acts->out_fc1,params->fc2.size.in_size, grad_acts->grad_out_fc2,
                 params->fc2.size.out_size,params->fc2.weights,params->fc2.bias, grad_acts->grad_out_fc1,mementun->mem_fc2,lr);

    fc_backward(acts->out_pool2, params->fc2.size.in_size, grad_acts->grad_out_fc1,params->fc1.size.out_size,params->fc1.weights, params->fc1.bias,
                grad_acts->grad_out_pool2, mementun->mem_fc1, lr);
    
    pool_backward(acts->out_conv2, params->pool2.size.in_size, grad_acts->grad_out_pool2, params->pool2.size.out_size,
                    grad_acts->grad_out_conv2, params->pool2.pool_size);
    
    conv_backward(acts->out_pool1, params->conv2.size.in_size, grad_acts->grad_out_conv2,params->conv2.size.out_size,
                    acts->out_conv2, grad_acts->grad_out_pool1, params->conv2.weights, mementun->mem_conv2, params->conv2.kernel_size,
                    params->conv2.stride, params->conv2.filters, lr);
    
    pool_backward(acts->out_conv1, params->pool1.size.in_size, grad_acts->grad_out_pool1,params->pool1.size.out_size,
                    grad_acts->grad_out_conv1, params->pool1.pool_size);
    
    conv_backward(inp, params->conv1.size.in_size, grad_acts->grad_out_conv1, params->conv1.size.out_size,acts->out_conv1, NULL,
                    params->conv1.weights, mementun->mem_conv1, params->conv1.kernel_size, params->conv1.stride,params->conv1.filters,lr);

}


int main(int argc, char const *argv[])
{

    clock_t start, end;
    srand(time(NULL));

    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);
    CNN model;
    CNN_init(&model, ImageSize, K1,C1,1,P1,K2,C2,1,P2,FC1_SIZE, OUTPUT_SIZE, BATCH);

    int train_size = (int)(dataloader.nImages*TRAIN_SPLIT);
    int test_size = dataloader.nImages - train_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        start = clock();
        
        for(int b=0;b<train_size/BATCH;b++){
            // float* images = dataloader.images + b*BATCH*ImageSize*ImageSize;
            load_betch_images(&dataloader, &model.datas, b, BATCH);
            float loss = 0.0f;
            float corr = 0.0f;
            for (int t = 0; t < BATCH; t++){
                float* images = model.datas.data + t*ImageSize*ImageSize;
                int label_idx = model.datas.labels[t];
                cnn_forward(&model,images,dataloader.imageSize.row,dataloader.imageSize.col);
                loss -= logf(model.acts.out_fc2[label_idx] + 1e-10f);
                // printf("label: %d, output: %f\n", label_idx, model.acts.output[label_idx]);
                cnn_backward(&model,images, label_idx, LEARN_RATE, model.params.fc2.size.out_size.z);
                corr += model.acts.out_fc2[label_idx]>0.5f?1.0f:0.0f;
            }
            
            // loss = 0.0f;
            printf("epoch: %d,batch:%d,  loss:%.3f  corr: %.3f \n", epoch,b, loss/BATCH, corr/BATCH);
        }
        

        float corr = 0.0f;
        for(int t=0;t<test_size;t++){    
            float* test_images = dataloader.images + (train_size+t)*ImageSize*ImageSize;
            int test_labels = dataloader.labels + train_size+t;
            cnn_forward(&model,test_images,dataloader.imageSize.row,dataloader.imageSize.col);
            corr += model.acts.out_fc2[test_labels]>0.5f?1.0f:0.0f;
        }
        printf("epoch: %d, test accuracy: %f\n", epoch, corr/test_size);
        end = clock();
    }
    

    DataLoader_clear(&dataloader);
    CNN_clear(&model);
    return 0;
}
