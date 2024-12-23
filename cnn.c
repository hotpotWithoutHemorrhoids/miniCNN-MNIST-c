#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <dataloader.h>
#include <float.h>


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

typedef struct{
    int x;
    int y;
    int z;
} Shape;

typedef struct{
    float* weights;
    int size;
    int stride;
    int filters;
} Conv;

typedef struct{
    float* weights;
    int input_size;
    int output_size;
}FC;

typedef struct{
    Conv conv1; // (K1,K1,C1)
    int pool1Size;
    Conv Conv2; // (K2,K2,C2)
    int pool2Size;
    // Conv Conv3; // (K3,K3,C3)
    FC fc;
    FC output;

}Paramerters;

typedef struct{
    float* conv1; // ( (imageSize-K1)/stride + 1, (imageSize-K1)/stride + 1, C1)
    Shape conv1_size;
    float* relu1; // ( (imageSize-K1)/stride + 1, (imageSize-K1)/stride + 1, C1)
    float* pool1; // ( ((imageSize-K1)/stride + 1)/P1, ((imageSize-K1)/stride + 1)/P1, C1)
    Shape pool1_size;
    float* conv2; // ( ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, C2)
    Shape conv2_size;
    float* relu2; // ( ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, ((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1, C2)
    float* pool2; // ( (((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1)/P2, (((imageSize-K1)/stride + 1)/P1 - K2)/stride + 1)/P2, C2)
    Shape pool2_size;
    float* fc; 
    Shape fc_size;
    float* output; // (1,1,10)
    Shape output_size;
}Activation;

typedef struct{
    float* grad_conv1;
    float* grad_pool1;
    float* grad_conv2;
    float* grad_pool2;
    float* grad_fc;
    float* grad_output;
}Grad_Activation;


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
    model->params.conv1.size = k1;
    model->params.conv1.stride = stride1;
    model->params.conv1.filters = c1;
    model->params.pool1Size = p1;
    model->params.Conv2.size = k2;
    model->params.Conv2.stride = stride2;
    model->params.Conv2.filters = c2;
    model->params.pool2Size = p2;
    model->params.fc.input_size = ((((imageSize - k1)/stride1 + 1)/p1 - k2)/stride2 + 1)*((((imageSize - k1)/stride1 + 1)/p1 - k2)/stride2 + 1)*c2;
    model->params.fc.output_size = fc1_size;
    model->params.output.input_size = fc1_size;
    model->params.output.output_size = outputSize;

    model->params_memory = NULL;
    model->acts_memory = NULL;
    model->grad_acts_memory = NULL;

    int total_params = k1*k1*c1 + k2*k2*c2 + model->params.fc.input_size*fc1_size + fc1_size*outputSize;
    model->toal_params = total_params;
    printf("total params: %d\n", total_params);
    model->params_memory = (float*)malloc(total_params * sizeof(float));
    // init weights
    model->params.conv1.weights = model->params_memory;
    init_params(model->params.conv1.weights, k1*k1*c1);
    model->params.Conv2.weights = model->params_memory + k1*k1*c1;
    init_params(model->params.Conv2.weights, k2*k2*c2);
    model->params.fc.weights = model->params_memory + k1*k1*c1 + k2*k2*c2;
    init_params(model->params.fc.weights, model->params.fc.input_size*model->params.fc.output_size);
    model->params.output.weights = model->params_memory + k1*k1*c1 + k2*k2*c2 + model->params.fc.input_size*fc1_size;
    init_params(model->params.output.weights, fc1_size*outputSize);

    model->datas.data = (float*)malloc(imageSize*imageSize*batch*sizeof(float));
    model->datas.labels = (int*)malloc(batch*sizeof(int));

    // init activation
    Activation* acts = &(model->acts);
    acts->conv1_size = (Shape){(imageSize - k1)/stride1 + 1, (imageSize - k1)/stride1 + 1, c1};
    acts->pool1_size = (Shape){acts->conv1_size.x/p1, acts->conv1_size.y/p1, c1};
    acts->conv2_size = (Shape){(acts->pool1_size.x - k2)/stride2 + 1, (acts->pool1_size.y - k2)/stride2 + 1, c2};
    acts->pool2_size = (Shape){acts->conv2_size.x/p2, acts->conv2_size.y/p2, c2};
    acts->fc_size = (Shape){1,1,fc1_size};
    acts->output_size = (Shape){1,1,outputSize};

    long int total_acts = 0;
    {
    total_acts += acts->conv1_size.x*acts->conv1_size.y*acts->conv1_size.z;
    total_acts += acts->pool1_size.x*acts->pool1_size.y*acts->pool1_size.z;
    total_acts += acts->conv2_size.x*acts->conv2_size.y*acts->conv2_size.z;
    total_acts += acts->pool2_size.x*acts->pool2_size.y*acts->pool2_size.z;
    total_acts += acts->fc_size.z;
    total_acts += acts->output_size.z;
    }
    model->total_acts = total_acts;
    printf("total_acts: %ld\n", total_acts);
    model->acts_memory = (float*)malloc(total_acts*sizeof(float));
    int offset = 0;
    acts->conv1 = model->acts_memory;
    offset += acts->conv1_size.x*acts->conv1_size.y*acts->conv1_size.z;
    acts->pool1 = model->acts_memory + offset;
    offset += acts->pool1_size.x*acts->pool1_size.y*acts->pool1_size.z;
    acts->conv2 = model->acts_memory + offset;
    offset += acts->conv2_size.x*acts->conv2_size.y*acts->conv2_size.z;
    acts->pool2 = model->acts_memory + offset;
    offset += acts->pool2_size.x*acts->pool2_size.y*acts->pool2_size.z;
    acts->fc = model->acts_memory + offset;
    offset += acts->fc_size.z;
    acts->output = model->acts_memory + offset;

    // init grad_acts
    if (model->grad_acts_memory == NULL){
        model->total_grad_acts = 0;
        model->total_grad_acts += outputSize;
        model->total_grad_acts += fc1_size;
        model->total_grad_acts += acts->pool2_size.x*acts->pool2_size.y*acts->pool2_size.z;
        model->total_grad_acts += acts->conv2_size.x*acts->conv2_size.y*acts->conv2_size.z;
        model->total_grad_acts += acts->pool1_size.x*acts->pool1_size.y*acts->pool1_size.z;
        model->total_grad_acts += acts->conv1_size.x*acts->conv1_size.y*acts->conv1_size.z;
        model->grad_acts_memory = (float*)malloc(model->total_grad_acts*sizeof(float));
        offset = 0;
        model->grad_acts.grad_conv1 = model->grad_acts_memory;
        offset += acts->conv1_size.x*acts->conv1_size.y*acts->conv1_size.z;
        model->grad_acts.grad_pool1 = model->grad_acts_memory + offset;
        offset += acts->pool1_size.x*acts->pool1_size.y*acts->pool1_size.z;
        model->grad_acts.grad_conv2 = model->grad_acts_memory + offset;
        offset += acts->conv2_size.x*acts->conv2_size.y*acts->conv2_size.z;
        model->grad_acts.grad_pool2 = model->grad_acts_memory + offset;
        offset += acts->pool2_size.x*acts->pool2_size.y*acts->pool2_size.z;
        model->grad_acts.grad_fc = model->grad_acts_memory + offset;
        offset += acts->fc_size.z;
        model->grad_acts.grad_output = model->grad_acts_memory + offset;
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
    #pragma omp parallel for
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


Shape fc_forward(float* inp, int inp_size, float* out, float* weights, int output_size){
    Shape output_shape = {1,1,output_size};

    for(int i = 0; i<output_size; i++){
        out[i] = 0.0f;
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
    Activation* acts = &model->acts;
    Shape conv1_shape = conv_forward(inp,h,w,1, acts->conv1, model->params.conv1.weights, model->params.conv1.size, model->params.conv1.stride, model->params.conv1.filters);
    Shape pool1_shape = pool_froward(acts->conv1, conv1_shape.x, conv1_shape.y, conv1_shape.z, acts->pool1, model->params.pool1Size);
    Shape conv2_shape = conv_forward(acts->pool1, pool1_shape.x, pool1_shape.y, pool1_shape.z, acts->conv2, model->params.Conv2.weights, model->params.Conv2.size, model->params.Conv2.stride, model->params.Conv2.filters);
    Shape pool2_shape = pool_froward(acts->conv2, conv2_shape.x, conv2_shape.y, conv2_shape.z, acts->pool2, model->params.pool2Size);
    // printf("pool2_shape: %d, %d, %d\n", pool2_shape.x, pool2_shape.y, pool2_shape.z);
    int flatten_size = pool2_shape.x*pool2_shape.y*pool2_shape.z;
    Shape fc_shape = fc_forward(acts->pool2, flatten_size, acts->fc, model->params.fc.weights, model->params.fc.output_size);
    Shape output_shape = fc_forward(acts->fc, fc_shape.z, acts->output, model->params.output.weights, model->params.output.output_size);
    softmax_forward(acts->output, output_shape.z);
}



void softmax_backward(float* inp, int inp_size, float* inp_grad){
    for(int i=0;i<inp_size;i++){
        inp[i] = inp[i]*(1-inp[i]);
    }
}

void cnn_backward(CNN *model, float lr){
    Activation acts = model->acts;
    Grad_Activation grad_acts = model->grad_acts;
    printf("backward: grad_acts is NULL: %d\n", model->grad_acts_memory == NULL);
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
            for (int t = 0; t < BATCH; t++){
                float* images = model.datas.data + t*ImageSize*ImageSize;
                int label_idx = model.datas.labels[t];
                cnn_forward(&model,images,dataloader.imageSize.row,dataloader.imageSize.col);
                loss += - logf(model.acts.output[label_idx] + 1e-10f);
                cnn_backward(&model, LEARN_RATE);
            }
            printf("epoch: %d, batch: %d, loss: %f\n", epoch, b, loss/BATCH);

        }

    }
    


    free(dataloader.images);
    free(dataloader.labels);
    free(model.params_memory);
    free(model.datas.data);
    free(model.datas.labels);
    free(model.acts_memory);
    free(model.grad_acts_memory);
    return 0;
}
