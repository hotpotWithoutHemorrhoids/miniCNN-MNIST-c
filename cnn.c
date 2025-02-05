#include<time.h>
#include<unistd.h>

#include "dataloader.h"
#include "debug.h"
#include "cnn.h"

// 1: relu, 2: leakRelu
#define THRESHOLD 0.6f
#define RELU 2
#if RELU == 2
    #define LEAK_RELU_SCALE 0.1f
#endif
#define LEARN_RATE 0.005f

// #define LOG true

void init_size(Size *size, int in_x,int in_y,int in_z,int out_x,int out_y,int out_z){
    size->in_size.x = in_x;
    size->in_size.y = in_y;
    size->in_size.z = in_z;
    size->out_size.x = out_x;
    size->out_size.y = out_y;
    size->out_size.z = out_z;
}

void init_conv(Conv* conv, int in_x,int in_y, int in_channel, int kernel_size, int stride, int filters){
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->filters = filters;
    init_size(&(conv->size), in_x, in_y, in_channel, 
                (in_x - kernel_size)/stride + 1, (in_y - kernel_size)/stride + 1, filters);
    conv->weights_size = filters* in_channel*kernel_size*kernel_size;
    conv->num_params = conv->weights_size;
    conv->weights = NULL;
}

void init_pool(Pool* pool, int in_x, int in_y, int in_z, int pool_size){
    init_size(&(pool->size), in_x, in_y, in_z, in_x/pool_size, in_y/pool_size, in_z);
    pool->pool_size = pool_size;
}

void init_fc(FC* fc, int in_size, int out_size){
    init_size(&(fc->size), 1, 1, in_size, 1, 1, out_size);
    fc->weight_size = in_size*out_size;
    fc->bias_size = out_size;
    fc->num_params = fc->weight_size + fc->bias_size;
    fc->weights = NULL;
    fc->bias = NULL;
}

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
        model->total_grad_acts = total_acts;

        model->acts_memory = (float*)malloc(total_acts*sizeof(float));
        model->grad_acts_memory = (float*)malloc(total_acts*sizeof(float));

        for(int i=0;i<total_acts;i++){
            model->acts_memory[i] = 0.0f;
            model->grad_acts_memory[i] = 0.0f;
        }

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
        for (int i = 0; i < total_mementun; i++){
            model->mementun_memory[i] = 0.0f;
        }
        
        model->mementun.mem_conv1 = model->mementun_memory + offset;
        offset += model->params.conv1.num_params;

        model->mementun.mem_conv2 = model->mementun_memory + offset;
        offset += model->params.conv2.num_params;

        model->mementun.mem_fc1 = model->mementun_memory + offset;
        offset += model->params.fc1.num_params;

        model->mementun.mem_fc2 = model->mementun_memory + offset;

    }
}

void initialize_memory(float* mem, int size){
    for(int i=0;i<size;i++){
        mem[i] = 0.0f;
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

Shape conv_forward(float *inp, Shape in_shape,float* out, 
                    float* conv_weights, int kernel_size,
                    int stride, int channel){
    /* 
    inp: (h,w,z)
    out: ((h-kernel_size)/stride+1,(w-kernel_size)/stride+1, channel)
    conv_weights: (kernel_size,kernel_size,channel)
    */
    // TODO
    int h = in_shape.y, w = in_shape.x,  in_channel = in_shape.z;
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
                for(int in_c=0 ;in_c<in_channel; in_c++){
                    float* inp_c = inp + in_c*h*w;
                    float* conv_weights_c = conv_weights + c*in_channel*kernel_size*kernel_size + in_c*kernel_size*kernel_size;
                    for (int k = 0; k < kernel_size; k++){
                        for (int l = 0; l < kernel_size; l++){
                            sum += inp_c[(i*stride+k)*w + j*stride+l]*conv_weights_c[k*kernel_size+l];
                        }
                    }
                }

                // relu
                #if RELU == 2
                    out[c*out_size + i*out_w + j] = sum> 0 ?sum : LEAK_RELU_SCALE*sum;
                #else
                    out[c*out_size + i*out_w + j] = sum>0?sum:0.0f;
                #endif

            }
        }
    }
    #ifdef LOG
    for(int i=0;i<channel; i++){
        char buff[50];
        snprintf(buff, sizeof(buff), "conv forward result, channel %d", i);
        printMatrix(out, out_h, out_w, buff);
    }
    printf("===================================\n");
    #endif
    return output_shape;
}

Shape pool_froward(float* inp, int h, int w, int z, float* out, int pool_size){
    int out_h = h/pool_size;
    int out_w = w/pool_size;
    Shape output_shape = {out_h, out_w, z};
    int out_size = out_h*out_w; 
    // printf("inp h:%d, w: %d, z:%d \n", h,w,z);
    // printVector(inp, h*w*z, "pool inp");

    // #pragma omp parallel for
    for (int c = 0; c < z; c++){
        float* inp_c = inp + c*h*w;
        // output (i,j) is map input pool (i*pool_size:i*pool_size+pool_size,j*pool_size:j*pool_size+pool_size)
        for (int i = 0; i < out_h; i++){
            for (int j = 0; j < out_w; j++){
                float max = -FLT_MAX;
                for (int k = 0; k < pool_size; k++){
                    for (int l = 0; l < pool_size; l++){
                        max = max > (inp_c[(i*pool_size+k)*w + j*pool_size+l])?max:inp_c[(i*pool_size+k)*w + j*pool_size+l];
                    }
                }
                out[c*out_size + i*out_w + j] = max;
            }
        }
    }

    #ifdef LOG
    for(int i=0;i<z; i++){
        char buff[50];
        snprintf(buff, sizeof(buff), "pool_froward result, channel %d", i);
        printMatrix(out, output_shapeout_h, out_w, buff);
    }
    printf("===================================\n");
    #endif
    // printf("out h:%d, w: %d, z:%d \n", out_h,out_w,z);
    // printVector(out, out_size*z, "pool out");

    return output_shape;
}


Shape fc_forward(float* inp, int inp_size, float* out, 
            float* weights, int output_size, float* bias, bool acti_func){
    /*    
        out = W * inp + bias
        W: (out_size, inp_size) , inp: inp_size
     */
    Shape output_shape = {1,1,output_size};

    for(int i = 0; i<output_size; i++){
        out[i] = bias[i];
    }

    for(int i=0; i<output_size; i++){
        for (int j = 0; j < inp_size; j++){
            out[i] += inp[j]*weights[i*inp_size+j];
        }
        if(acti_func){
            #if RELU == 2
                out[i] = out[i]> 0 ?out[i] : LEAK_RELU_SCALE*out[i];
            #else
                out[i] = out[i]>0?out[i] : 0;
            #endif
        }
    }

    #ifdef LOG

    char buff[50] = "fc_forward result";
    printVector(out, output_size, buff);

    printf("===================================\n");
    #endif
    return output_shape;
}

void softmax_forward(float* inp, int inp_size){
    float sum = 0.0f;
    float max = -FLT_MAX;
    for(int i=0;i<inp_size;i++){
        max = max>inp[i]?max:inp[i];
    }
    for(int i=0;i<inp_size;i++){
        inp[i] = expf(inp[i]-max);
        sum += inp[i];
    }
    float inv_sum = sum!=0.0f? 1.0f/(sum+EPSILON) :1.0f;
    for (int i = 0; i < inp_size; i++){
        inp[i] *= inv_sum;
    }

    #ifdef LOG
    printf("softmax sum: %f, inp_size: %d, max: %f\n", sum, inp_size, max);
    printVector(inp, inp_size, "softmax_forward");
    printf("===================================\n");
    #endif

    return;
}
   
void cnn_forward(CNN *model, float* inp, int h, int w){
    Activation* acts = &(model->acts);
    Paramerters* params = &(model->params);
    // printShape(&(params->conv1.size.in_size), "conv1 in shape");
    // printShape(&(params->conv1.size.out_size), "conv1 out shape");
    // printVector(inp, params->conv1.size.in_size.x*params->conv1.size.in_size.y*params->conv1.size.in_size.z, "inp->conv1");

    // printf("conv1 kernel size: %d, stride: %d, fliters: %d \n", params->conv1.kernel_size, params->conv1.stride, params->conv1.filters);
    // printVector(params->conv1.weights, params->conv1.weights_size, "conv1 weights");
    conv_forward(inp,params->conv1.size.in_size, 
                acts->out_conv1, params->conv1.weights, params->conv1.kernel_size, 
                params->conv1.stride, params->conv1.filters);
    // printVector(acts->out_conv1, params->conv1.size.out_size.x*params->conv1.size.out_size.y*params->conv1.size.out_size.z, "acts->out_conv1");

    
    
    pool_froward(acts->out_conv1,
                params->conv1.size.out_size.y,params->conv1.size.out_size.x, params->conv1.size.out_size.z,
                acts->out_pool1, params->pool1.pool_size);
    conv_forward(acts->out_pool1, params->conv2.size.in_size,
                acts->out_conv2, params->conv2.weights, params->conv2.kernel_size,params->conv2.stride, params->conv2.filters);
    pool_froward(acts->out_conv2, params->conv2.size.out_size.y, params->conv2.size.out_size.x,params->conv2.size.out_size.z,
                acts->out_pool2,params->pool2.pool_size);
    // printf("pool2_shape: %d, %d, %d\n", pool2_shape.x, pool2_shape.y, pool2_shape.z);
    // int flatten_size = pool2_shape.x*pool2_shape.y*pool2_shape.z;
    fc_forward(acts->out_pool2, params->pool2.size.out_size.x*params->pool2.size.out_size.y*params->pool2.size.out_size.z,
                acts->out_fc1, params->fc1.weights, params->fc1.size.out_size.z,params->fc1.bias, true);
    // fc_forward(acts->fc, fc_shape.z, acts->output, model->params.output.weights, model->params.output.output_size);
    // printVector(acts->out_fc1, params->fc1.size.out_size.z, "fc1 output");
    fc_forward(acts->out_fc1, params->fc1.size.out_size.z, 
                acts->out_fc2, params->fc2.weights, params->fc2.size.out_size.z,params->fc2.bias, true);
    printVector(acts->out_fc2, params->fc2.size.out_size.z, "forward before softmax");
    softmax_forward(acts->out_fc2, params->fc2.size.out_size.z);
    printVector(acts->out_fc2, params->fc2.size.out_size.z, "forward after softmax");
} 

void softmax_backward(float* inp, int inp_size,int target, float* d_inp){
    for(int i=0;i<inp_size;i++){
        d_inp[i] = inp[i] - (i==target?1.0f:0.0f);
    }
}

void fc_backward(float* inp, Shape inp_size, float* d_loss,
                Shape out_size,float* out, float* weights, float* bias,
                float* d_inp, float* mementun, float lr, bool acti_func){
    /* 
        weights: (inp_len, out_len) 
    */
   int inp_len = inp_size.x*inp_size.y*inp_size.z;
   int out_len = out_size.z; // fc 输出1维

    // RELU backward
    if(acti_func){
        for(int i=0;i<out_len;i++){
            #if RELU == 2
                d_loss[i] *= out[i]> 0 ?1 : LEAK_RELU_SCALE;
            #else
                d_loss[i] *= out[i]>0?1:0.0f;
            #endif
        }
    }

    // update d_inp
    for (int i = 0; i < out_len; i++){
        for (int j = 0; j < inp_len; j++){
            d_inp[j] += weights[i*inp_len + j]*d_loss[i];
        }
    }

    float* weight_mementun = mementun;
    float* bias_mementun = mementun + inp_len*out_len;

    // update weights
    for (int i = 0; i < out_len; i++){
        float* weight_row = weights + i*inp_len;
        float* mementun_row = weight_mementun + i*inp_len;
        for (int j = 0; j < inp_len; j++){
            float grad_w_ij = d_loss[i] * inp[j];
            mementun_row[j] = mementun_row[j] * MOMENTUM + lr*grad_w_ij;
            weight_row[j] -= mementun_row[j]; 

            // weight_row[j] -=  lr*grad_w_ij;
            
            // test
            // mementun_row[j] = grad_w_ij;
        }    
    }
    
    for(int i=0;i<out_len; i++){
        bias_mementun[i] = bias_mementun[i]*MOMENTUM + lr*d_loss[i];
        bias[i] -= bias_mementun[i];
        
        // test
        // bias_mementun[i] = d_loss[i];
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
                float max = -FLT_MAX;
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
    // TODO update conv backward
    int out_h = out_size.x;
    int out_w = out_size.y;
    int out_z = out_size.z;
    int inp_h = inp_size.x;
    int inp_w = inp_size.y;
    int inp_z = inp_size.z;
    // int inp_z = inp_size.z;
    // relu backward
    for(int z = 0;z<out_z;z++){
        for (int x = 0; x < out_h; x++){
            for (int y = 0; y < out_w; y++){
                #if RELU ==2
                    d_loss[z*out_h*out_w + x*out_w + y] *= out[z*out_h*out_w + x*out_w + y]>0?1.0f: LEAK_RELU_SCALE;
                #else
                    d_loss[z*out_h*out_w + x*out_w + y] *= out[z*out_h*out_w + x*out_w + y]>0?1.0f:0.0f;
                #endif
            }
        }}
    
    // update mementun
/*     for(int c=0;c<channel;c++){
        for(int i=0;i<kernel_size;i++){
            // float* conv_weights_row = conv_weights + c*kernel_size*kernel_size + i*kernel_size;
            for(int j=0;j<kernel_size;j++){
                float grad_w_ij = 0.0f;
                for (int inp_c = 0; inp_c < inp_size.z; inp_c++){
                    float* mementun_row = mementun +c*inp_c*kernel_size*kernel_size+ inp_c*kernel_size*kernel_size + i*kernel_size;
                    float* inp_c_image = inp + inp_c*inp_h*inp_w;
                    for(int l=0;l<out_size.x; l++){
                        for(int k=0;k<out_size.y;k++){
                                grad_w_ij += inp_c_image[(i*stride+l)*out_w+j*stride+k]*d_loss[c*out_h*out_w + l*out_w + k];
                        }
                    }
                // mementun_row[j] = mementun_row[j]*MOMENTUM + lr*grad_w_ij;
                mementun_row[j] = grad_w_ij;

                // conv_weights_row[j] += mementun_row[j];
            }
        }
        }
        
    } */
   for (int c = 0; c < out_z; c++){
        // 第c个conv kernel
        for(int inp_c=0;inp_c<inp_z;inp_c++){
            // inp_c's matrix in the conv kernel map to inp_c's inp channel
            float* inp_channel = inp + inp_c*inp_h*inp_w;
            float* mementun_c = mementun + c*inp_z*kernel_size*kernel_size + inp_c*kernel_size*kernel_size;
            float* d_loss_c = d_loss + c*out_h*out_w;

            for(int i=0;i<kernel_size; i++){
                for(int j=0;j<kernel_size;j++){
                    // kernel loc (i,j)

                    /* TODO 对比一下看哪种顺序效率更快
                    for(int inp_c=0;inp_c<inp_z;inp_c++){
                        // inp_c's matrix in the conv kernel map to inp_c's inp channel
                        float* inp_channel = inp + inp_c*inp_h*inp_w;
                        float* mementun_c = mementun + c*inp_z*kernel_size*kernel_size + inp_c*kernel_size*kernel_size;
                        float* d_loss_c = d_loss + c*out_h*out_w; */

                        float grad_w_ij = 0.0f;
                        for(int l=0; l<out_h; l++){
                            for(int k=0; k<out_w; k++){
                                grad_w_ij += inp_channel[(l+i)*inp_w+(k+j)]*d_loss_c[l*out_w+k];
                            }
                        }
                        mementun_c[i*kernel_size+j] = mementun_c[i*kernel_size+j]*MOMENTUM+ lr*grad_w_ij;
                        // mementun_c[i*kernel_size+j] = grad_w_ij;
                        
                        // mementun_c[i*kernel_size+j] = lr*grad_w_ij;
                        
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
        int new_row = out_size.x+2*(kernel_size-1), new_col = out_size.y+2*(kernel_size-1); 
        float* full_conv_dloss = (float*)malloc(new_row*new_col*sizeof(float));
        for(int i=0;i<new_row*new_col; i++){
            full_conv_dloss[i] = 0.0f;
        }

        for(int z=0;z<out_z;z++){
            float* d_loss_z = d_loss + z*out_h*out_w;
            // 第z个卷积核的权重
            float* conv_weights_z = conv_weights + z*inp_z*kernel_size*kernel_size;
            // full model padding
           for (int i = 0; i < out_h; i++){
                for (int j = 0; j < out_w; j++){
                    full_conv_dloss[(i+kernel_size-1)*new_col + (j+kernel_size-1)] = d_loss_z[i*out_w+j];
                }
           }
           
           for(int inp_c=0; inp_c<inp_z; inp_c++){
                float* conv_weights_z_inpc = conv_weights_z + inp_c*kernel_size*kernel_size;
                float* d_inp_c = d_inp + inp_c*inp_h*inp_w;
                for(int i=0;i<inp_h; i++){
                    for(int j=0;j<inp_w; j++){
                        float d_inp_ij = 0.0f;

                        for(int l=0;l<kernel_size;l++){
                            for (int k = 0; k < kernel_size; k++){
                                d_inp_ij += full_conv_dloss[(i+l)*new_col + (j+k)]*
                                            conv_weights_z_inpc[(kernel_size-l-1)*kernel_size + (kernel_size-k-1)];
                            }
                        }
                        d_inp_c[i*inp_w+j] += d_inp_ij;
                    }
                }
           }
        }
        free(full_conv_dloss);
    }

    // update weights
    for (int out_c = 0; out_c < out_z; out_c++){
        for(int c=0;c<inp_z;c++){
            for(int i=0;i<kernel_size;i++){
                float* mementun_row = mementun +out_c*inp_z*kernel_size*kernel_size + c*kernel_size*kernel_size + i*kernel_size;
                float* conv_weights_row = conv_weights + out_c*inp_z*kernel_size*kernel_size + c*kernel_size*kernel_size + i*kernel_size;
                for(int j=0;j<kernel_size;j++){
                    conv_weights_row[j] -= mementun_row[j];
                }
            }
        }
    }
}

void cnn_backward(CNN *model,float* inp,int label, float lr, int output_size){
    Activation* acts = &(model->acts);
    Grad_Activation* grad_acts = &(model->grad_acts);
    Paramerters* params = &(model->params);
    Mementun* mementun = &(model->mementun);
    float* output = acts->out_fc2;
    softmax_backward(output, output_size,label, grad_acts->grad_out_fc2);

    // printVector(grad_acts->grad_out_fc2, output_size, "grad_out_fc2");

    fc_backward(acts->out_fc1,params->fc2.size.in_size, grad_acts->grad_out_fc2,
                 params->fc2.size.out_size,acts->out_fc2 ,params->fc2.weights,params->fc2.bias, 
                 grad_acts->grad_out_fc1,mementun->mem_fc2,lr, true);

    fc_backward(acts->out_pool2, params->fc2.size.in_size, grad_acts->grad_out_fc1,
                params->fc1.size.out_size,acts->out_fc1, params->fc1.weights, params->fc1.bias,
                grad_acts->grad_out_pool2, mementun->mem_fc1, lr, true);
    
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

int main(int argc, char const *argv[]){ 

    clock_t start, end;
    srand(time(NULL));

    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);
    CNN model;
    CNN_init(&model, ImageSize, K1,C1,1,P1,K2,C2,1,P2,FC1_SIZE, OUTPUT_SIZE, BATCH);

    int train_size = (int)(dataloader.nImages*TRAIN_SPLIT);
    int test_size = dataloader.nImages - train_size;
    int row = dataloader.imageSize.row, col = dataloader.imageSize.col;
    int input_size = row * col;
    printf("train_size: %d, test_size: %d\n", train_size, test_size);
    int epoch = 0;
    float* image = (float*)malloc(INPUT_SIZE * sizeof(float));
    for (; epoch < EPOCHS; epoch++){
        /* for(int b=0;b<train_size/BATCH;b++){
            start = clock();
            // float* images = dataloader.images + b*BATCH*ImageSize*ImageSize;
            load_betch_images(&dataloader, &model.datas, b, BATCH);
            float loss = 0.0f;
            float corr = 0.0f;
            for (int t = 0; t < BATCH; t++){
                float* images = model.datas.data + t*ImageSize*ImageSize;
                // printMatrix(images, ImageSize,ImageSize, "image");
                int label_idx = model.datas.labels[t];
                // printf("label: %d\n", label_idx);
                cnn_forward(&model,images,dataloader.imageSize.row,dataloader.imageSize.col);
                
                loss -= logf(model.acts.out_fc2[label_idx] + 1e-10f);
                // printf("label: %d, output: %.2f\n", label_idx, model.acts.out_fc2[label_idx]);
                // printVector(model.acts.out_fc2, 10, "out_fc2");
                // TODO backward maybe question
                initialize_memory(model.grad_acts_memory, model.total_grad_acts);
                cnn_backward(&model,images, label_idx, LEARN_RATE, model.params.fc2.size.out_size.z);
                corr += model.acts.out_fc2[label_idx]>0.5f?1.0f:0.0f;
                // break;
            }
            end = clock();
                // loss = 0.0f;
            printf(" batch:%d,  loss:%.3f  corr: %.3f  cost time: %.3f\n", b, loss/BATCH, corr/BATCH, (float)(end-start)/CLOCKS_PER_SEC);
            // break;
        }

    float corr = 0.0f;
    for(int t=0;t<test_size;t++){    
        float* test_images = dataloader.images + (train_size+t)*ImageSize*ImageSize;
        int test_label_idx =  dataloader.labels[train_size+t];
        cnn_forward(&model,test_images,dataloader.imageSize.row,dataloader.imageSize.col);
        // printf("label: %d, output-prob: %f\n", test_label_idx, model.acts.out_fc2[test_label_idx]);
        corr += model.acts.out_fc2[test_label_idx]>0.5f?1.0f:0.0f;
    }
    
    printf("epoch: %d test accuracy: %f\n", epoch, corr/test_size); */

        // train
        float loss = 0, train_corr = 0.0f;
        start = clock();
        for (int i = 0; i < train_size; i++){
            
            // load a image
            for (int j = 0; j < input_size; j++){
                image[j] = (float) dataloader.images[i*input_size + j] / 255.0f;
            }
            int label = dataloader.labels[i];
            printf("i: %d, label: %d\n", i, label);

            cnn_forward(&model, image, row, col);
            loss -= logf(model.acts.out_fc2[label]);
            initialize_memory(model.grad_acts_memory, model.total_grad_acts);
            cnn_backward(&model,image, label, LEARN_RATE, model.params.fc2.size.out_size.z);
            train_corr += model.acts.out_fc2[label]> THRESHOLD ? 1 : 0;
            sleep(1);
        }
        end = clock();
        float train_time = (float)(end - start)/CLOCKS_PER_SEC;
        printf("epoch: %d, train time:%.2f, loss: %.2f, train_corr: %.2f \n",
                epoch, train_time, loss, train_corr);
        
        float test_corr = 0.0f;
        for (int i = 0; i < test_size; i++){
            // load a test image
            for (int j = 0; j < input_size; j++){
                image[j] = dataloader.images[(i+train_size)* input_size + j];
            }
            int test_label = dataloader.labels[i+train_size];

            cnn_forward(&model, image, row, col);
            test_corr += model.acts.out_fc2[test_label] > THRESHOLD? 1 : 0;
        }
        
        printf("epoch: %d, test corr:%.2f \n", epoch, test_corr/test_size);
    }
    free(image);

    DataLoader_clear(&dataloader);
    CNN_clear(&model);
    return 0;
}
