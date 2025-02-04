#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void fc1(float* inp,int inp_size, 
        float* weight, int out_size,float*out, float* bias){
    for (int i = 0; i < out_size; i++) {
        out[i] = bias[i];
    }

    for (int j = 0; j < inp_size; j++){
        for (int i = 0; i < out_size; i++){
            out[i] += weight[i*inp_size + j] * inp[j];
        }
    }

    for (int i = 0; i < out_size; i++){
        out[i] *= (out[i] > 0)?1.0f : 0.0f;
    }
}

void fc2(float* inp,int inp_size, 
        float* weight, int out_size,float*out, float* bias){
    
    for (int i = 0; i < out_size; i++){
        out[i] = bias[i];
        for (int j = 0; j < inp_size; j++){
            out[i] += weight[i*inp_size + j] * inp[j];
        }
        
        out[i] *= out[i]>0 ? 1.0f : 0.0f;
    }
}

void rand_init(float* arr, int size){
    // avg init arr in [-1, 1]
    for (int i = 0; i < size; i++){
        arr[i] = ((float)rand()/RAND_MAX - 0.5f)*2;
    }
    
}

int main(int argc, char const *argv[])
{

    int inp_size = 784, out_size = 256, run_time = 40000;
    float* inp = (float*)malloc(inp_size * sizeof(float));
    float* weight = (float*)malloc(inp_size * out_size * sizeof(float));
    float* bias = (float*)malloc(out_size*sizeof(float));
    float* out = (float*)malloc(out_size*sizeof(float));
    
    clock_t t1, t2,t3;
    float fc1_run_time = 0.0f, fc2_run_time = 0.0f;
    for (int t = 0; t < run_time; t++){
        rand_init(inp, inp_size);
        rand_init(weight, inp_size * out_size);
        rand_init(bias, out_size);
        t1 = clock();
        fc1(inp, inp_size,weight, out_size, out, bias);
        t2 = clock();
        fc2(inp, inp_size,weight, out_size, out, bias);
        t3 = clock();
        fc1_run_time += t2 - t1;
        fc2_run_time += t3 - t2;
    }
    printf("fc1 cost %.2f, fc2 cost: %.2f \n", 
                fc1_run_time/CLOCKS_PER_SEC, fc2_run_time/CLOCKS_PER_SEC);
    
    free(inp);
    free(weight);
    free(bias);
    free(out);
    return 0;
}


