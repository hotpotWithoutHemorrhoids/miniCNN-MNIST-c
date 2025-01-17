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
            out[i] = out[i]>0?out[i] : 0;
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
            // mementun_row[j] = mementun_row[j] * MOMENTUM + lr*grad_w_ij;
            // weight_row[j] -= mementun_row[j]; 
            
            // test
            mementun_row[j] = grad_w_ij;
        }    
    }
    
    for(int i=0;i<out_len; i++){
        // bias_mementun[i] = bias_mementun[i]*MOMENTUM + lr*d_loss[i];
        // bias[i] -= bias_mementun[i];
        
        // test
        bias_mementun[i] = d_loss[i];
    }
}


#define EPSILON 0.0005
void test_fc_forward(){
    float inp[] = {-0.7502,  0.3037, -0.1002,  1.6871,  0.8657,  0.8073,  1.1017,  1.8793,
         0.1578,  0.1991,  2.3571,  1.5748,  2.0154,  0.9386,  0.7626,  1.5392,
        -0.0075,  1.6734,  0.3956,  0.0998,  0.7317,  0.7323,  1.2466,  1.2966,
         1.5663,  2.5581,  1.8606};
    
    int inp_size =  27;
    int out_size = 2;
    float weights[] = {1.5496,  0.3476,  0.0930,  0.6147,  0.7124, -1.7765,  0.3539,  1.1996,
        -0.7123, -0.6200, -0.2281, -0.7893, -1.6111, -1.8716,  0.5431,  0.6607,
         0.2705,  0.5597, -0.3184,  1.5117, -1.3633, -0.9832,  1.5113,  0.6419,
        -0.7474, -0.9234,  0.5734, -0.1093,  0.5181,  0.1065,  0.2692,  1.3248,
         0.0375, -0.6378, -0.8148, -0.6895,  0.4175, -0.2127,  0.5269,  1.6193,
        -0.9640,  0.1415, -0.1637, -0.3582, -0.0594, -2.4919,  0.2389,  1.3440,
         0.1032,  1.1004, -0.3417,  0.9473,  0.6223, -0.4481 };
    float bias[] = {1.6268, -1.2522};
    float target_out[] = {-2.672312,  4.0622};
    float target_relu_out[2];
    float  fc_out[2];

    for (size_t i = 0; i < out_size; i++){
        target_relu_out[i] = target_out[i]>0?target_out[i]:0;
    }
    
    // printf("w:\n");
    // printMatrix(weights, 2,27, "weights");
    // printf("\n");
    // printVector(inp, 27, "inp_x");
    // printVector(bias, 2, "bias");

    
    fc_forward(inp, inp_size, fc_out, weights,out_size,bias,0);
    CU_ASSERT_TRUE(compare_float_arr(fc_out, target_out, out_size, EPSILON));

    // printVector(fc_out, out_size, "fc_out");
    // printVector(target_out, out_size, "target_out");

    fc_forward(inp, inp_size,fc_out, weights,out_size,bias,1);
    CU_ASSERT_TRUE(compare_float_arr(fc_out, target_relu_out, out_size, EPSILON));

    // printVector(fc_out, out_size, "fc_out");
    // printVector(target_relu_out, out_size, "target_relu_out");
}

void test_fc_backward(){
    Shape in_shape = {3,3,3};
    int inp_size = in_shape.x * in_shape.y*in_shape.z;
    float inp[] = {-0.7502,  0.3037, -0.1002,  1.6871,  0.8657,  0.8073,  1.1017,  1.8793,
         0.1578,  0.1991,  2.3571,  1.5748,  2.0154,  0.9386,  0.7626,  1.5392,
        -0.0075,  1.6734,  0.3956,  0.0998,  0.7317,  0.7323,  1.2466,  1.2966,
         1.5663,  2.5581,  1.8606};
    float fc_weight[] = { 1.5496,  0.3476,  0.0930,  0.6147,  0.7124, -1.7765,  0.3539,  1.1996,
        -0.7123, -0.6200, -0.2281, -0.7893, -1.6111, -1.8716,  0.5431,  0.6607,
         0.2705,  0.5597, -0.3184,  1.5117, -1.3633, -0.9832,  1.5113,  0.6419,
        -0.7474, -0.9234,  0.5734, -0.1093,  0.5181,  0.1065,  0.2692,  1.3248,
         0.0375, -0.6378, -0.8148, -0.6895,  0.4175, -0.2127,  0.5269,  1.6193,
        -0.9640,  0.1415, -0.1637, -0.3582, -0.0594, -2.4919,  0.2389,  1.3440,
         0.1032,  1.1004, -0.3417,  0.9473,  0.6223, -0.4481};
    float fc_bias[] = {1.6268, -1.2522};
    float d_loss_fc_out[] = {-0.9988,  0.9988};
    Shape out_shape = {1,1,2};
    float d_inp[27] = {0.0f}, mementum[2*27 + 2] = {0.0f};

    float target_d_inp[] = {-1.6570,  0.1703,  0.0135, -0.3451,  0.6117,  1.8118, -0.9905, -2.0120,
         0.0227,  1.0363,  0.0154,  1.3146,  3.2265,  0.9066, -0.4011, -0.8234,
        -0.6280, -0.6184, -2.1710, -1.2713,  2.7041,  1.0852, -0.4104, -0.9824,
         1.6928,  1.5439, -1.0203};
    float target_d_weight[] = {0.7493, -0.3034,  0.1000, -1.6851, -0.8647, -0.8063, -1.1004, -1.8771,
        -0.1576, -0.1988, -2.3543, -1.5729, -2.0130, -0.9375, -0.7617, -1.5374,
         0.0075, -1.6714, -0.3951, -0.0997, -0.7308, -0.7314, -1.2451, -1.2951,
        -1.5645, -2.5551, -1.8584, -0.7493,  0.3034, -0.1000,  1.6851,  0.8647,
         0.8063,  1.1004,  1.8771,  0.1576,  0.1988,  2.3543,  1.5729,  2.0130,
         0.9375,  0.7617,  1.5374, -0.0075,  1.6714,  0.3951,  0.0997,  0.7308,
         0.7314,  1.2451,  1.2951,  1.5645,  2.5551,  1.8584, -0.9988,  0.9988};
    

    fc_backward(inp, in_shape,d_loss_fc_out,  out_shape, fc_weight, fc_bias, d_inp, mementum, LEARN_RATE,false);

    CU_ASSERT_TRUE(compare_float_arr(d_inp, target_d_inp, inp_size, EPSILON));

    CU_ASSERT_TRUE(compare_float_arr(mementum, target_d_weight, 2*27 + 2, EPSILON));

    
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

