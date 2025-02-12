#include<stdlib.h>
#include<time.h>
#include<math.h>
#include <omp.h>

#include "dataloader.h"
#include "debug.h"

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"
#define EPOCHS 20
#define BATCH 1000
#define FC1_SIZE 256
#define OUTPUT_SIZE 10
#define TRAIN_SPLIT 0.8
#define THRESDHOLD 0.65f
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

inline void _init_params(float* p, unsigned int size){
    float scale = sqrtf(2.0f/ size);
    for (unsigned int i = 0; i < size; i++){
        p[i] = ((float)rand()/RAND_MAX - 0.5f)*scale*2;
    }
}

void NN_init(NN* nn, int input_size, int fc1_size, int output_size){
    nn->input_size = input_size;
    nn->fc1_size = fc1_size;
    nn->output_size = output_size;
    nn->total_params = input_size*fc1_size+fc1_size + fc1_size*output_size+output_size;
    nn->total_acts = fc1_size + output_size*2;
    nn->total_grad_acts = fc1_size + output_size;

    nn->params_mem = (float*)malloc(nn->total_params * sizeof(float));
    nn->momentum_memory = (float*)malloc(nn->total_params * sizeof(float));
    if(nn->params_mem != NULL){
        memset(nn->params_mem, 0.0f, nn->total_params);
        memset(nn->momentum_memory, 0.0f, nn->total_params);

        nn->params.fc1_weights = nn->params_mem;
        nn->momentum.fc1_weight_mom = nn->momentum_memory;

        nn->params.fc1_bias = nn->params_mem +  input_size*fc1_size;
        nn->momentum.fc1_bias_mom = nn->momentum_memory + input_size*fc1_size;

        nn->params.fc2_weights = nn->params_mem +  input_size*fc1_size+fc1_size;
        nn->momentum.fc2_weight_mom = nn->momentum_memory + input_size*fc1_size+fc1_size;

        nn->params.fc2_bias = nn->params_mem +  input_size*fc1_size+fc1_size + fc1_size*output_size;
        nn->momentum.fc2_bias_mom = nn->momentum_memory + input_size*fc1_size+fc1_size + fc1_size*output_size;

        // init param
        _init_params(nn->params.fc1_weights, input_size*fc1_size);
        _init_params(nn->params.fc2_weights, fc1_size*output_size);
        
    }

    nn->acts_memory = (float*)malloc(nn->total_acts * sizeof(float));
    nn->grad_acts_memory = (float*)malloc(nn->total_grad_acts * sizeof(float));
    if(nn->acts_memory != NULL){
        nn->acts.fc1_output = nn->acts_memory;
        nn->acts.fc2_output = nn->acts_memory + fc1_size;
        nn->acts.output = nn->acts_memory + fc1_size + output_size;

        memset(nn->acts_memory, 0, nn->total_acts);
    }

    if(nn->grad_acts_memory != NULL){
        nn->grad_acts.grad_fc1_output = nn->grad_acts_memory;
        nn->grad_acts.grad_fc2_output = nn->grad_acts_memory + fc1_size;

        memset(nn->grad_acts_memory, 0, nn->total_grad_acts);
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
}


void fc_forward(float* inp, int inp_size, 
                float* out, int out_size, float* weight, float* bias){
    
    // #pragma omp parallel for
    // for (int i = 0; i < out_size; i++){
    //     out[i] = bias[i];
    //     float* weights_row = weight + i*inp_size;

    //     for(int j=0; j<inp_size; j++){
    //         out[i] += inp[j] * weights_row[j];
    //     }

    //     // leak_relu
    //     out[i] *= out[i] >0 ? 1.0f : LEAK_RELU_SCALE;
    // }

    for (int i = 0; i < out_size; i++) {
        out[i] = bias[i];
    }

    // TODO ？ 这里有个疑问， 为什么将inp_size 作为外层循环的时候会更快，快了将进45%
    // 将outsize作为外层的时候，理论上访问weight应该是连续的，
    // 而将inp_size放于外层的时候weight应该是不连续的才是
    // #pragma omp parallel for
    for (int j = 0; j < inp_size; j++){
        for (int i = 0; i < out_size; i++){
            out[i] += weight[i*inp_size + j] * inp[j];
        }
    }

    for (int i = 0; i < out_size; i++){
        out[i] *= (out[i] > 0)?1.0f : LEAK_RELU_SCALE;
    }
}

void softmax_forward(float* inp, int size, float* output){
    float max = inp[0], sum = 0.0f;
    for(int i=1; i<size; i++){
        if(inp[i] > max) max = inp[i];
    }

    for (int i = 0; i < size; i++){
        output[i] = expf(inp[i] - max);
        sum += output[i];
    }

    sum = sum==0.0f?1e-10f : sum;
    for (int i = 0; i < size; i++){
        output[i] /= sum;
    }
}

void nn_forward(NN *nn, float* inp, float* count_time){
    // nn_forward
    int inp_size = nn->input_size, fc1_size = nn->fc1_size, output_size=nn->output_size;
    if (count_time != NULL){
        double t1, t2, t3;

        t1 = omp_get_wtime();
        fc_forward(inp, inp_size, 
                nn->acts.fc1_output, fc1_size,
                 nn->params.fc1_weights, nn->params.fc1_bias);
        t2 = omp_get_wtime();
        fc_forward(nn->acts.fc1_output, fc1_size,
                nn->acts.fc2_output, output_size,
                nn->params.fc2_weights, nn->params.fc2_bias);
        t3 = omp_get_wtime();
        count_time[0] += (float)(t2 - t1);
        count_time[1] += (float)(t3 - t2);
        softmax_forward(nn->acts.fc2_output, output_size, nn->acts.output);
    }else{

        fc_forward(inp, inp_size, 
                nn->acts.fc1_output, fc1_size,
                 nn->params.fc1_weights, nn->params.fc1_bias);
        fc_forward(nn->acts.fc1_output, fc1_size,
                nn->acts.fc2_output, output_size,
                nn->params.fc2_weights, nn->params.fc2_bias);
        softmax_forward(nn->acts.fc2_output, output_size, nn->acts.output);
    }
}

int nn_predict(NN* nn, float* inp){

    nn_forward(nn, inp, NULL);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (nn->acts.output[i] > nn->acts.output[max_index])
            max_index = i;

    return max_index;
}

void softmax_backward(float* output, int size, 
                        int target_label, float* grad_acts){
    for (int i = 0; i < size; i++){
        grad_acts[i] = output[i] - (i == target_label? 1.0f : 0.0f);
    }
}

void fc_backward(float* inp, int input_size,float* out, float* grad_output, int output_size,
                 float* weights, float* bias, float* weight_mom, float* bias_mom,float lr, float* d_inp, int is_relu){
    
    // relu backward
    if(is_relu){
        for(int i=0;i<output_size; i++){
            grad_output[i] *= out[i] >0?1.0f:LEAK_RELU_SCALE;
        }
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

    softmax_backward(acts->output,output_size, target_label,grad_acts->grad_fc2_output);

    fc_backward(acts->fc1_output, fc1_size,acts->fc2_output,grad_acts->grad_fc2_output, output_size,
                params->fc2_weights, params->fc2_bias, 
                momentums->fc2_weight_mom, momentums->fc2_bias_mom, lr,grad_acts->grad_fc1_output, 0);
    fc_backward(inp, input_size, acts->fc1_output,grad_acts->grad_fc1_output,fc1_size,
                params->fc1_weights,params->fc1_bias,momentums->fc1_weight_mom, momentums->fc1_bias_mom,lr,
                NULL,1);
}

int main(int argc, char const *argv[])
{
    double start, end;
    srand(time(NULL));
    omp_set_num_threads(4); 

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
    double t1, t2, t3;
    
    float* images = (float*)malloc(input_size * sizeof(float));
    for(;epoch<EPOCHS; epoch++){
        start = omp_get_wtime();
        float loss = 0.0f;
        float train_correct=0;
        float test_correct=0;
        
        float forward_time=0, back_time=0;
        float count_time[2] = {0};
        for (int i = 0; i < train_size; i++){
            // load a image
            for (int j = 0; j < input_size; j++){                
                images[j] = (float)(dataloader.images[i*input_size + j]) / 255.0f;

            }
            int target_label = (int)dataloader.labels[i];
            t1 = omp_get_wtime();
            
            nn_forward(&nn, images, count_time);
            t2 = omp_get_wtime();
            loss -= logf(nn.acts.output[target_label] + 1e-10f);
            train_correct += nn.acts.output[target_label]>THRESDHOLD?1:0;
            nn_backward(&nn, images, target_label, LEARNING_RATE);
            t3 = omp_get_wtime();
            forward_time += (t2 - t1);
            back_time += (t3 - t2);
        }
        t1 = omp_get_wtime();
        // break;
        printf("loss: %f, train correct: %.3f, train cost: %.2fs\n", 
                loss, train_correct/train_size, (t1 - start));
        printf("train process forward cost: %.2f s, backward cost: %.2fs \n",
                forward_time, back_time);
        printf("forward:  fc1 cost: %.2f, fc2 cost: %.2f \n",
                 count_time[0], count_time[1]);

        for (int i = 0; i < test_size; i++){
            for (int j = 0; j < input_size; j++){
                images[j] = (float)(dataloader.images[(i+train_size)*input_size + j]) / 255.0f;
            }
            int test_label = (int)dataloader.labels[i+train_size];
            if (nn_predict(&nn, images) == test_label){
                test_correct++;
            }
        }
        
        end = omp_get_wtime();
        double cost_time = (end -start);
        printf("epoch: %d, test accuracy: %.3f, train set avg_loss: %.4f, train&test cost time: %.3f s\n",
                epoch, (float)test_correct/test_size, loss/train_size, cost_time);
    }

    free(images);
    DataLoader_clear(&dataloader);
    NN_clear(&nn);
    return 0;
}
