#include<stdlib.h>
#include<time.h>
#include<math.h>

#include "dataloader.h"

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"
#define EPOCHS 10
#define BATCH 1000
#define FC1_SIZE 256
#define OUTPUT_SIZE 10
#define TRAIN_SPLIT 0.8
#define THRESDHOLD 0.5f
#define MOMENTUM 0.9f
#define LEAK_RELU_SCALE 0.2f
#define LEARNING_RATE 0.0005f

typedef struct
{
    float* fc1_weights;
    float* fc1_bias;
    float* fc2_weights;
    float* fc2_bias;
}Param;

typedef struct{
    float* fc1_weight_mom;
    float* fc1_bias_mom;
    float* fc2_weight_mom;
    float* fc2_bias_mom;
}Momentum;

typedef struct 
{
    float* fc1_output;
    float* fc2_output;
    float* output;
}Act;

typedef struct 
{
    float* grad_fc1_output;
    float* grad_fc2_output;
}Grad;

typedef struct 
{
    Data datas;
    int input_size;
    float* params_mem;
    int total_params;
    Param params;
    float* momentum_memory;
    Momentum momentum;
    float* acts_memory;
    int total_acts;
    Act acts;
    float* grad_acts_memory;
    int total_grad_acts;
    Grad grad_acts;
    int fc1_size;
    int output_size;
}NN;

void NN_init(NN* nn, int input_size, int fc1_size, int output_size){
    nn->input_size = input_size;
    nn->fc1_size = fc1_size;
    nn->output_size = output_size;
    nn->total_params = input_size*fc1_size+fc1_size + fc1_size*output_size+output_size;
    nn->total_acts = fc1_size + output_size*2;
    nn->total_grad_acts = fc1_size + output_size;
    nn->datas.data = NULL, nn->datas.labels = NULL;

    nn->params_mem = (float*)malloc(nn->total_params * sizeof(float));
    nn->momentum_memory = (float*)malloc(nn->total_params * sizeof(float));
    if(nn->params_mem != NULL){
        nn->params.fc1_weights = nn->params_mem;
        nn->momentum.fc1_weight_mom = nn->momentum_memory;

        nn->params.fc1_bias = nn->params_mem +  input_size*fc1_size;
        nn->momentum.fc1_bias_mom = nn->momentum_memory + input_size*fc1_size;

        nn->params.fc2_weights = nn->params_mem +  input_size*fc1_size+fc1_size;
        nn->momentum.fc2_weight_mom = nn->momentum_memory + input_size*fc1_size+fc1_size;

        nn->params.fc2_bias = nn->params_mem +  input_size*fc1_size+fc1_size + fc1_size*output_size;
        nn->momentum.fc2_bias_mom = nn->momentum_memory + input_size*fc1_size+fc1_size + fc1_size*output_size;
    }

    nn->acts_memory = (float*)malloc(nn->total_acts * sizeof(float));
    nn->grad_acts_memory = (float*)malloc(nn->total_grad_acts * sizeof(float));
    if(nn->acts_memory != NULL){
        nn->acts.fc1_output = nn->acts_memory;
        nn->acts.fc2_output = nn->acts_memory + fc1_size;
        nn->acts.output = nn->acts_memory + fc1_size + output_size;
    }

    if(nn->grad_acts_memory != NULL){
        nn->grad_acts.grad_fc1_output = nn->grad_acts_memory;
        nn->grad_acts.grad_fc2_output = nn->grad_acts_memory + fc1_size;
    }

    // TODO： 初始化 Data
    if(nn->datas.data == NULL && nn->datas.labels == NULL){
        printf("init datas %d \n",nn->input_size * BATCH);
        nn->datas.data = (float*)malloc(nn->input_size * BATCH * sizeof(float));
        nn->datas.labels = (int*)malloc(BATCH * sizeof(int));
    }   

}

void NN_clear(NN* nn){
    if(nn->params_mem != NULL){
        free(nn->params_mem);
    }
    if(nn->momentum_memory){
        free(nn->momentum_memory);
    }
    if(nn->acts_memory != NULL){
        free(nn->acts_memory);
    }
    if(nn->grad_acts_memory != NULL){
        free(nn->grad_acts_memory);
    }

    if(nn->datas.data != NULL){
        free(nn->datas.data);
        free(nn->datas.labels);
    }
}


void fc_forward(float* inp, int inp_size, 
                float* out, int out_size, float* weight, float* bias){

    for (int i = 0; i < out_size; i++){
        float* weights_row = weight + i*inp_size;
        out[i] = bias[i];
        for(int j=0; j<inp_size; j++){
            out[i] += inp[j] * weights_row[j];
        }

        // leak_relu
        out[i] = out[i] >0 ? out[i] : LEAK_RELU_SCALE*out[i];
    }
}

void softmax_forward(float* inp, int size, float* output){
    float max = 0.0f, sum = 0.0f;
    for(int i=0; i<size; i++){
        if(inp[i] > max) max = inp[i];
    }

    for (int i = 0; i < size; i++){
        output[i] = expf(inp[i] - max);
        sum += output[i];
    }

    sum = sum==0.0f?1e-8f : sum;
    for (int i = 0; i < size; i++){
        output[i] /= sum;
    }
}

void nn_forward(NN *nn, float* inp){
    // TODO nn_forward
    int inp_size = nn->input_size, fc1_size = nn->fc1_size, output_size=nn->output_size;
    fc_forward(inp, inp_size, 
                nn->acts.fc1_output, fc1_size,
                 nn->params.fc1_weights, nn->params.fc1_bias);
    fc_forward(nn->acts.fc1_output, fc1_size,
                nn->acts.fc2_output, output_size,
                nn->params.fc2_weights, nn->params.fc2_bias);
    softmax_forward(nn->acts.fc2_output, output_size, nn->acts.output);
}

void softmax_backward(float* output, int size, 
                        int target_label, float* grad_acts){
    for (int i = 0; i < size; i++){
        grad_acts[i] = output[i] - (i == target_label? 1.0f : 0.0f);
    }
}

void fc_backward(float* inp, int input_size,float* out, float* grad_output, int output_size,
                 float* weights, float* bias, float* weight_mom, float* bias_mom,float lr, float* d_inp){
    
    // relu backward
    for(int i=0;i<output_size; i++){
        grad_output[i] *= out[i] >0?1.0f:LEAK_RELU_SCALE;
    }

    if(d_inp){
        // d_inp = W^T * grad_out where W is (output_size, input_size)
        for(int i=0;i<input_size; i++){
            d_inp[i] = 0.0f;
        }
        for (int j = 0; j < output_size; j++){
            float* weight_row = weights + j *input_size;
            for(int i=0; i<input_size; i++){
                d_inp[i] += grad_output[j] * weight_row[i];
            }
        }
    }

    // update weights
    for (int i = 0; i < output_size; i++){
        float* weight_mom_row = weight_mom + i*input_size;
        float* weight_row = weights + i*input_size;
        for (int j = 0; j < input_size; j++){
            weight_mom_row[j] = weight_mom_row[j]*MOMENTUM+  lr*grad_output[i]*inp[j];
            weight_row[j] -=  weight_mom_row[j];
        }   
    }
    // update bias
    for (int i = 0; i < output_size; i++){
        bias_mom[i] = bias_mom[i]*MOMENTUM + lr * grad_output[i];
        bias[i] -= bias_mom[i];
    }
}

void nn_backward(NN* nn, float* inp, int target_label, float lr){
    // TODO nn_backward
    Act* acts = &(nn->acts);
    Grad* grad_acts = &(nn->grad_acts);
    Param* params =&(nn->params);
    Momentum* momentums = &(nn->momentum);
    int input_size= nn->input_size, fc1_size = nn->fc1_size, output_size=nn->output_size;

    softmax_backward(acts->fc2_output,output_size, target_label,grad_acts->grad_fc2_output);
    fc_backward(acts->fc1_output, fc1_size,acts->fc2_output,grad_acts->grad_fc2_output, output_size,
                params->fc2_weights, params->fc2_bias, 
                momentums->fc2_weight_mom, momentums->fc2_bias_mom, lr,grad_acts->grad_fc1_output);
    fc_backward(inp, input_size, acts->fc1_output,grad_acts->grad_fc1_output,fc1_size,
                params->fc1_weights,params->fc1_bias,momentums->fc1_weight_mom, momentums->fc1_bias_mom,lr,
                NULL);
}

int main(int argc, char const *argv[])
{
    clock_t start, end;
    srand(time(NULL));

    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);
    NN nn;
    int input_size = dataloader.imageSize.col * dataloader.imageSize.row;
    NN_init(&nn,input_size, FC1_SIZE, OUTPUT_SIZE);

    int train_size = (int)(dataloader.nImages*TRAIN_SPLIT);
    int test_size = dataloader.nImages - train_size;
    printf("train_size: %d, test_size: %d\n", train_size, test_size);
    printf("input sdize: %d, fc1:size: %d, output_size: %d, batch: %d \n",nn.input_size,nn.fc1_size, nn.output_size, BATCH);
    
    int epoch = 0;
    for(;epoch<EPOCHS; epoch++){
        for(int b=0;b<train_size/BATCH;b++){
        start = clock();
        load_betch_images(&dataloader, &(nn.datas), b, BATCH);
        float loss = 0.0f, corr = 0.0f;
        for(int t = 0; t<BATCH; t++){
            float* images = nn.datas.data + t*input_size;
            int label_idx = nn.datas.labels[t];
            // printf("label: %d\n", label_idx);
            nn_forward(&nn, images);

            loss -= logf(nn.acts.output[label_idx] + 1e-10f);
            corr += nn.acts.output[label_idx]>THRESDHOLD ?1.0f:0.0f;
            nn_backward(&nn, images, label_idx, LEARNING_RATE);
        }
        end = clock();
        printf(" batch:%d,  loss:%.3f  corr: %.3f  cost time: %.3f\n", b, loss/BATCH, corr/BATCH, (float)(end-start)/CLOCKS_PER_SEC);
        }

        unsigned int correct_num = 0;
        for(int i=0; i<test_size; i++){
            float* test_images = (float*)(dataloader.images + (train_size+i)*input_size);
            int test_label = dataloader.labels[train_size+i];
            nn_forward(&nn,test_images);
            correct_num += nn.acts.output[test_label]>THRESDHOLD? 1 : 0;
        }
        printf("epoch: %d test accuracy: %.4f\n", epoch, (float)(correct_num/test_size));

    }
    DataLoader_clear(&dataloader);
    NN_clear(&nn);
    return 0;
}
