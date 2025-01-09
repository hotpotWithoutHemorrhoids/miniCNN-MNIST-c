/* 
test fc_forward and fc_backward in cnn.c 
*/
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn.h"
#include "test_util.h"
#include "debug.h"


Shape fc_forward(float* inp, int inp_size, float* out, 
            float* weights, int output_size, float* bias, bool acti_func){
    Shape output_shape = {1,1,output_size};

    for(int i = 0; i<output_size; i++){
        out[i] += bias[i];
    }

    for(int i=0;i<inp_size; i++){
        for(int j=0;j<output_size;j++){
            out[j] += inp[i]*weights[i*output_size+j];
        }
    }
    
    // RELU
    if(acti_func){
        for(int i=0;i<output_size;i++){
            out[i] = out[i]>0?out[i]:0.0f;
        }
    }

    return output_shape;
}


void fc_backward(float* inp, Shape inp_size, float* d_loss,
                Shape out_size, float* weights, float* bias,
                float* d_inp, float* mementun, float lr, bool acti_func){
    /* 
        weights: (inp_len, out_len) 
    */
   int inp_len = inp_size.x*inp_size.y*inp_size.z;
   int out_len = out_size.z; // fc 输出1维

    // RELU backward
    if(acti_func){
        for(int i=0;i<out_len;i++){
            d_loss[i] *= d_loss[i]>0?1:0.0f;
        }
    }

    // update d_inp
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
            mementun_row[j] = mementun_row[j]*MOMENTUM + lr*gradW_ij;
            weight_row[j] -= mementun_row[j];
        }
    }

    for(int i=0;i<out_len; i++){
        bias_mementun[i] = bias_mementun[i]*MOMENTUM + lr*d_loss[i];
        bias[i] -= bias_mementun[i];
    }
}


#define EPSILON 0.00001
void test_fc_forward(){
    float a[16] = {0.43, 0.05, -2.38,-0.75, 1.51, -0.6 ,1.37,1.09,1.11, 0.09,-1.26,0.21,0.31,0.4 , 1.13,-0.22};
    float w[16*10] = {0.43, -1.02,  0.1 , -0.7 , -0.21, -0.03,  0.43, -0.29, -1.65,
        0.23, -2.  , -1.72, -1.91,  0.39, -0.08, -0.83, -0.24, -0.19,
        0.14, -0.67, -0.18, -0.19,  1.1 , -0.99,  0.77,  0.36, -1.05,
       -0.97, -0.52, -1.17,  0.27, -0.8 , -1.11,  1.52, -0.09, -0.68,
       -1.2 ,  1.21, -1.12,  0.3 ,  0.84,  0.05, -0.94, -0.29, -2.22,
        1.09,  0.08, -1.08,  0.76, -0.57,  1.79, -0.44, -1.74,  0.44,
        0.43,  0.38,  0.2 ,  0.45,  0.21, -1.51,  0.16,  0.3 ,  1.81,
        0.99,  1.43, -1.05, -0.38,  0.  ,  0.72,  0.07,  0.42,  3.13,
       -0.46,  0.13, -0.28,  0.13,  1.14,  1.29,  0.95, -0.66, -0.46,
       -1.59,  0.49,  0.56,  0.18,  0.39,  1.9 ,  0.63,  0.06, -0.42,
       -0.83,  0.56, -0.74,  1.39,  1.6 ,  1.63, -0.68, -0.66,  0.72,
        0.27, -1.22, -0.27,  0.22,  0.7 ,  0.  , -0.34,  2.5 , -0.7 ,
        0.44,  1.2 ,  0.42, -0.17, -0.46, -1.33,  2.17, -1.33,  0.3 ,
        1.37,  1.24,  0.85,  0.45,  0.23, -2.5 ,  0.82, -0.49,  0.02,
        0.79, -0.66,  0.07, -1.92,  0.15,  1.42, -0.13, -1.05,  1.91,
       -1.42,  0.08,  1.81, -0.73, -0.37, -0.55, -1.42, -0.49, -0.06,
        0.96, -0.82, -1.53,  0.92,  1.49,  0.42,  0.46,  1.15,  1.16,
       -0.08, -0.6 , -1.91,  0.17, -0.52, -0.87,  0.6 };
    float bias[10] = {-0.23,  0.22, -0.91, -0.74,  0.24, -1.2 , -0.08,  1.22, -0.37,-0.05};
    float target_out[10] = {1.4691,  2.2815, -2.6775,  0.3602, -0.9473, -1.8196,  1.6861,
        5.4752,  5.4908, -0.1792};
    float target_relu_out[10];
    for (size_t i = 0; i < 10; i++){
        target_relu_out[i] = target_out[i]>0?target_out[i]:0;
    }
    
    float fc_out[10] = {0.0f};
    printf("w:\n");
    printMatrix(w, 16,10, "w");
    printf("\n");
    
    fc_forward(a, 16,fc_out, w,10,bias,0);
    CU_ASSERT_TRUE(compare_float_arr(fc_out, target_out, 10, EPSILON));

    fc_forward(a, 16,fc_out, w,10,bias,1);
    CU_ASSERT_TRUE(compare_float_arr(fc_out, target_relu_out, 10, EPSILON));
    printf("fc_out\n");
    printVector(fc_out, 10, "fc_out");
    printf("target_out\n");
    printVector(target_relu_out, 10, "target_out");
}

void test_fc_backward(){

}

// 初始化测试套件
int init_suite(void) {
    return 0;
}

// 清理测试套件
int clean_suite(void) {
    return 0;
}


// 编译：  gcc fc_layer_test.c -o fc_layer_test -I../include -lcunit
int main(){   
    // 初始化
    if(CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();

    CU_pSuite suite = CU_add_suite("fc_layer test", init_suite, clean_suite);
    if(suite == NULL){
        CU_cleanup_registry();
        return CU_get_error();
    }

    // 添加测试用例到测试套件中
    CU_add_test(suite, "test_fc_forward", test_fc_forward);
    CU_add_test(suite, "test fc_backward", test_fc_backward);

    // 运行所有测试用例
    CU_basic_set_mode(CU_BRM_VERBOSE); 
    CU_basic_run_tests();

    // 清理测试框架
    CU_cleanup_registry();

    return 0;
}

