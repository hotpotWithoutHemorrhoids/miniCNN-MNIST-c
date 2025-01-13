#include<CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "test_util.h"
#include "debug.h"
#define EPSILON 0.001

Shape conv_forward(float *inp, Shape in_shape,
                    float* out, float* conv_weights, 
                    int kernel_size, int stride, int channel){
    /* 
    inp: (h,w,z)
    out: ((h-kernel_size)/stride+1,(w-kernel_size)/stride+1, channel)
    conv_weights: (kernel_size,kernel_size,channel)
    */
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
                out[c*out_size + i*out_w + j] = sum>0?sum:0.0f;
            }
        }
    }
    return output_shape;
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
                d_loss[z*out_h*out_w + x*out_w + y] *= out[z*out_h*out_w + x*out_w + y]>0?1.0f:0.0f;
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
                        // mementun_c[i*kernel_size+j] = mementun_c[i*kernel_size+j]*MOMENTUM+ lr*grad_w_ij;
                        mementun_c[i*kernel_size+j] = grad_w_ij;
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
        for(int c=0;c<out_z;c++){
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

void test_conv_forward(){
    float x[] = {-0.3495,  0.6522,  0.4267,  0.0836, -0.2726, -2.1014,  0.3729, -1.0535,
         0.9384};
    float kernel_weight[] = {0.6017,  1.7250, -0.8483, -1.3722};
    float target_out[] = {1.2179,  4.2432,  0.7093, 0};
    Shape in_shape = {3,3,1};
    Shape out_shape = {2,2,1};
    float conv_out[4] = {0};
    conv_forward(x, in_shape, conv_out, kernel_weight, 2, 1,1);
    CU_ASSERT_TRUE(compare_float_arr(conv_out, target_out, 4, EPSILON));
    printVector(conv_out, 4,"");
}

void test_conv_forward_1(){
    float x[] = {0.6614,  0.2669,  0.0617,  0.6213, -0.4519, -0.1661, -1.5228,  0.3817,
        -1.0276};
    float kernel_weight[] = {-0.5631, -0.8923, -0.0583, -0.1955};
    float target_out[] = {0.0000, 0.0000, 0.0675, 0.5814};
    Shape in_shape = {3,3,1};
    Shape out_shape = {2,2,1};
    float conv_out[4] = {0};
    conv_forward(x, in_shape, conv_out, kernel_weight, 2, 1,1);
    CU_ASSERT_TRUE(compare_float_arr(conv_out, target_out, 4, EPSILON));
    printVector(conv_out, 4,"");
}

void test_conv_backward(){
    Shape inp_shape={3,3,1};
    float x[] = {0.6614,  0.2669,  0.0617,  0.6213, -0.4519, -0.1661, -1.5228,  0.3817,
    -1.0276};
    float kernel_weight[] = {-0.5631, -0.8923, -0.0583, -0.1955};
    float d_kernel_weight[4] = {0};
    float d_loss[] = {0.0000,  0.0000, -0.3551,  3.6442};
    Shape out_shape = {2,2,1};
    float target_out[] = {0.0000, 0.0000, 0.0675, 0.5814};
    float d_inp[9] = {0};
    
    int inp_len=inp_shape.x*inp_shape.y*inp_shape.z, kernel_len = 4;
    float target_dinp[] = {0.0000,  0.0000,  0.0000, 0.1999, -1.7350, -3.2517, 0.0207, -0.1429, -0.7125};
    float target_dkernel_weight[] = {-1.8675, -0.4449, 1.9317, -3.8804};

    int kernel_size=2, stride=1, out_channel=1;
    conv_backward(x, inp_shape, d_loss, out_shape, 
                target_out,d_inp, kernel_weight,d_kernel_weight,
                kernel_size,stride,out_channel,LEARN_RATE);
    CU_ASSERT_TRUE(compare_float_arr(d_inp, target_dinp, inp_len,EPSILON));

    CU_ASSERT_TRUE(compare_float_arr(d_kernel_weight, target_dkernel_weight, kernel_len,EPSILON));
    // printVector(d_kernel_weight, kernel_len, "d_kernel_weight");
    // printVector(target_dkernel_weight, kernel_len, "target_dkernel_weight");

}


// 2 inpt channel, 2 output channel
void test_conv_backward_1(){
    Shape inp_shape={3,3,2};
    float x[] = {-0.9414,  1.2632,  0.0031, -0.1535,  1.1396, -0.2302,  1.1877,  0.7677,
        -0.7117,  0.4349, -0.8558, -0.2346, -0.4215,  0.8488, -0.6776, -0.9445,
         2.1698, -1.1736};
    float kernel_weight[] = {2.3693,  0.2829, -0.2345,  1.6892,  0.2716, -0.1365, -0.7042,  2.0126,
        -0.9694,  0.6403,  0.8201, -0.9151, -2.1437,  1.4072,  1.2744, -0.1874};
    float d_kernel_weight[16] = {0};
    float d_loss[8] = {0.0350, -0.2541,  0.0306,  0.0000,  0.0000, -0.1980,  0.0999, -0.3076};
    Shape out_shape = {2,2,2};
    float target_out[] = {2.3282,  0.1759,  5.7787, -2.3141, -2.2803,  2.6358,  1.6378,  0.2407};
    float d_inp[18] = {0};
    
    int inp_len=inp_shape.x*inp_shape.y*inp_shape.z, kernel_len = 16;
    float target_dinp[] = {0.0830, -0.4003, -0.1987, -0.0326,  0.3272, -0.4450,  0.0748, -0.2921,
         0.2815,  0.0095,  0.3507, -0.2440, -0.2305,  0.7930, -0.9073,  0.1058,
        -0.3493,  0.0576};

    float target_dkernel_weight[] = {-0.3587,  0.0783, -0.2587,  0.1219,  0.2198,  0.0556, -0.2594,  0.2682,
        -0.6161,  0.1841, -0.3432,  0.3412, -0.1338,  0.3397, -0.9300,  0.7120};

    int kernel_size=2, stride=1, out_channel=2;
    
    conv_backward(x, inp_shape, d_loss, out_shape, 
                target_out,d_inp, kernel_weight,d_kernel_weight,
                kernel_size,stride,out_channel,LEARN_RATE);

    CU_ASSERT_TRUE(compare_float_arr(d_inp, target_dinp, inp_len,EPSILON));
    printVector(d_inp, inp_len, "d_inp");
    printVector(target_dinp, inp_len, "target_dinp");
    printf("============\n");
    CU_ASSERT_TRUE(compare_float_arr(d_kernel_weight, target_dkernel_weight, kernel_len,EPSILON));
    // printVector(d_kernel_weight, kernel_len, "d_kernel_weight");
    // printVector(target_dkernel_weight, kernel_len, "target_dkernel_weight");

}


// 初始化测试套件
int init_suite(void) {
    return 0;
}

// 清理测试套件
int clean_suite(void) {
    return 0;
}


// 编译：  gcc conv_test.c -o conv_test -I../include -lcunit
int main(){   
    // 初始化
    if(CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();

    CU_pSuite suite = CU_add_suite("conv_layer test", init_suite, clean_suite);
    if(suite == NULL){
        CU_cleanup_registry();
        return CU_get_error();
    }

    // 添加测试用例到测试套件中
    CU_add_test(suite, "test_conv_forward", test_conv_forward);
    CU_add_test(suite, "test_conv_forward1", test_conv_forward_1);
    CU_add_test(suite, "test_conv_backward", test_conv_backward);
    
    CU_add_test(suite, "test_conv_backward_1", test_conv_backward_1);

    // 运行所有测试用例
    CU_basic_set_mode(CU_BRM_VERBOSE); 
    CU_basic_run_tests();

    // 清理测试框架
    CU_cleanup_registry();

    return 0;
}
