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

void test_conv_forward_2(){
    float inp[] = {0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.09,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.17,0.18,0.45,0.45,0.52,0.72,0.97,0.62,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.94,0.95,0.89,0.89,0.89,0.89,0.89,0.89,1.00,1.00,1.00,0.96,0.94,0.98,1.00,0.51,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03,0.76,1.00,1.00,0.78,0.77,0.77,0.67,0.50,0.45,0.22,0.22,0.07,0.00,0.86,0.95,0.05,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.35,0.76,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.56,1.00,0.62,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03,0.88,1.00,0.11,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.31,1.00,0.71,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.73,1.00,0.26,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.22,1.00,0.87,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.62,1.00,0.41,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.13,1.00,0.82,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.73,1.00,0.36,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.26,0.98,0.78,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.78,0.98,0.24,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.45,1.00,0.69,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.14,0.88,0.95,0.08,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.56,1.00,0.31,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.31,0.99,0.56,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.11,0.95,0.63,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.07,0.82,0.62,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.55,0.63,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00};
    float kernel_weight[] = {-0.3113, -0.7130, -0.7291, -0.2992, -0.2529, -0.3602,  0.9394,  1.1614,
        -0.1706,  0.5119,  0.5962,  1.2911,  1.7541, -0.4149, -0.9922, -0.2986,
         0.6443, -0.2710, -0.1359,  2.5745, -0.5229,  0.9863,  0.2923,  1.0146,
         1.5558};
    // this target out is relu result of conv result
    float target_out[] = {0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.4002e-01, 2.4689e-01, 1.2777e-01, 1.1800e-01,
        5.1569e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.6449e-01, 4.5253e-01,
        9.3243e-01, 1.3770e+00, 1.4858e+00, 2.1290e+00, 2.6002e+00, 2.4368e+00,
        1.3508e+00, 7.6145e-01, 1.0429e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 7.7790e-02, 1.7449e+00, 2.6916e+00, 2.6346e+00, 3.4972e+00,
        3.0308e+00, 2.8990e+00, 2.9602e+00, 2.9602e+00, 3.5690e+00, 3.6833e+00,
        4.3631e+00, 4.4795e+00, 4.4130e+00, 5.1196e+00, 5.6629e+00, 3.9896e+00,
        1.2463e+00, 8.6489e-01, 8.9937e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.7540e-01, 3.5368e+00, 4.5036e+00, 4.8782e+00, 6.0496e+00,
        5.3139e+00, 4.9491e+00, 4.5319e+00, 4.2712e+00, 4.1114e+00, 3.4591e+00,
        3.0921e+00, 2.9240e+00, 2.9017e+00, 4.4887e+00, 5.3113e+00, 3.6719e+00,
        3.8709e+00, 3.4845e+00, 9.6135e-01, 1.9121e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 2.7625e-02, 1.0608e+00, 1.8087e+00, 4.3161e+00, 4.8234e+00,
        4.7422e+00, 4.5669e+00, 3.2067e+00, 3.2393e+00, 3.1416e+00, 2.4684e+00,
        2.9604e+00, 2.9621e+00, 3.9398e+00, 7.7644e+00, 7.7752e+00, 5.1179e+00,
        6.0426e+00, 4.5723e+00, 9.1204e-01, 6.5808e-02, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.2078e+00, 3.5077e+00,
        4.2192e+00, 4.3573e+00, 3.5956e+00, 3.8886e+00, 3.8407e+00, 3.5902e+00,
        3.2514e+00, 2.7511e+00, 5.0330e+00, 5.6333e+00, 2.3805e+00, 2.6513e+00,
        3.7118e+00, 1.3244e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 2.7120e-03, 1.2132e-01, 0.0000e+00, 0.0000e+00, 5.3494e-01,
        1.7689e+00, 8.6723e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.5860e+00, 1.8277e+00, 0.0000e+00, 2.2795e+00,
        3.2800e+00, 7.8098e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.2039e+00, 3.7575e+00, 1.8780e+00, 2.4146e+00, 4.5343e+00,
        2.0299e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        3.9377e-01, 3.3635e+00, 4.0852e+00, 1.1411e+00, 3.1852e+00, 4.0445e+00,
        3.9031e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.5558e-02,
        1.5213e+00, 4.1509e+00, 2.8554e+00, 1.7153e+00, 4.0823e+00, 2.5742e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.2800e-01,
        3.0745e+00, 4.0243e+00, 1.2300e+00, 3.0597e+00, 4.2165e+00, 9.2811e-01,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4605e+00,
        4.3415e+00, 2.7799e+00, 1.4792e+00, 3.7876e+00, 2.9707e+00, 3.9047e-02,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0451e-01, 3.5440e+00,
        3.9730e+00, 1.0415e+00, 2.8602e+00, 3.9071e+00, 1.0193e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8829e+00, 4.1435e+00,
        2.4350e+00, 1.5838e+00, 3.8141e+00, 2.8321e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 7.0011e-01, 3.7625e+00, 3.8972e+00,
        1.0806e+00, 3.1383e+00, 3.9865e+00, 8.4433e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.1781e-01, 2.6697e+00, 4.2843e+00, 1.9789e+00,
        1.8650e+00, 3.7816e+00, 2.2533e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.2317e+00, 3.9240e+00, 3.1035e+00, 8.2052e-01,
        3.2713e+00, 3.5744e+00, 6.5986e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 4.8230e-01, 3.1576e+00, 3.7638e+00, 8.4948e-01, 2.5006e+00,
        3.8244e+00, 1.7360e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.7114e-01, 2.3877e+00, 3.9988e+00, 1.3990e+00, 1.3306e+00, 3.1090e+00,
        2.5535e+00, 1.2051e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0891e-01,
        1.6300e+00, 3.9403e+00, 1.5723e+00, 5.3378e-01, 2.5290e+00, 2.8409e+00,
        8.5057e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0359e+00,
        3.5306e+00, 1.4362e+00, 2.7080e-03, 2.2352e+00, 2.7599e+00, 1.5253e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00};
    
    Shape in_shape = {28,28,1};
    Shape out_shape = {24,24,1};
    float* conv_out = (float*)malloc(24*24*1*sizeof(float));
    conv_forward(inp, in_shape, conv_out, kernel_weight, 5, 1, 1);
    CU_ASSERT_TRUE(compare_float_arr(conv_out, target_out, 24*24*1, EPSILON));
    // printVector(conv_out, 4,"");
    free(conv_out);
}

void test_conv_forward_3(){
    float inp[] = {0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.09,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.17,0.18,0.45,0.45,0.52,0.72,0.97,0.62,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.94,0.95,0.89,0.89,0.89,0.89,0.89,0.89,1.00,1.00,1.00,0.96,0.94,0.98,1.00,0.51,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03,0.76,1.00,1.00,0.78,0.77,0.77,0.67,0.50,0.45,0.22,0.22,0.07,0.00,0.86,0.95,0.05,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.35,0.76,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.56,1.00,0.62,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03,0.88,1.00,0.11,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.31,1.00,0.71,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.73,1.00,0.26,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.22,1.00,0.87,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.62,1.00,0.41,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.13,1.00,0.82,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.73,1.00,0.36,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.26,0.98,0.78,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.78,0.98,0.24,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.45,1.00,0.69,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.14,0.88,0.95,0.08,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.56,1.00,0.31,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.31,0.99,0.56,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.11,0.95,0.63,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.07,0.82,0.62,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.55,0.63,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00};
    float kernel_weight[] = {-0.3113, -0.7130, -0.7291, -0.2992, -0.2529, -0.3602,  0.9394,  1.1614,
        -0.1706,  0.5119,  0.5962,  1.2911,  1.7541, -0.4149, -0.9922, -0.2986,
         0.6443, -0.2710, -0.1359,  2.5745, -0.5229,  0.9863,  0.2923,  1.0146,
         1.5558};
    float target_out[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.14002122323568683,0.24689678827978948,0.12777495588946747,0.11799764081746275,0.05156937262018635,-0.05228502891747725,0.0,0.0,0.2644845327785196,0.4525313434681765,0.9324397833423667,1.3769801982977832,1.485796331350216,2.129040868599383,2.6002112793827292,2.436851235899611,1.350874494383301,0.7614718672836062,0.10431579110514907,-0.3241671792883589,0.0,0.0,
0.0,0.07778956846427047,1.7448850716969087,2.6916028729315356,2.634695761967844,3.4972417111159833,3.0308971073447943,2.899072620357013,2.9603037221755413,2.9603037221755413,3.569113623095003,3.6833630568498172,4.363217732814353,4.479604484809666,4.413116887936525,5.119743766742469,5.663007644037227,3.9896764681279957,1.2463716037638242,0.8649572909361601,0.08999248064180815,-0.45178617118099285,0.0,0.0,
0.0,0.17540104998975825,3.536818126254584,4.503626596946395,4.878266421490472,6.049653111541602,5.314008441744109,4.949266466994163,4.532081011345287,4.271296895263683,4.111535068051975,3.4592366061875754,3.0922057902215205,2.9240671796922664,2.9017813101240453,4.488756157299815,5.311390671653451,3.6720113324161807,3.870923055903286,3.484598660205941,0.9614442273846073,0.19122720164874057,0.0,0.0,
0.0,0.027625322931754185,1.060776220133487,1.808700446954458,4.316042938246608,4.823426406753825,4.742218406736592,4.566909494710957,3.206728572564365,3.239262737628568,3.141614722339767,2.4683843064229025,2.9604266008214433,2.962085055178264,3.9398368988584833,7.764456769576425,7.775303146841786,5.117979691185618,6.042670346338123,4.572321878102473,0.9120918718136841,0.06583034872693835,0.0,0.0,
0.0,-0.004169479692528656,-0.2908528055333987,-0.02333821225912395,3.2078201787492113,3.5077254792756465,4.219224311002753,4.357420125313696,3.5956679179471993,3.8886948795472938,3.8407940793209816,3.5902916872377624,3.2514485739369428,2.7511661755463064,5.033063290121187,5.6334277981181184,2.380593708345182,2.6513130426344045,3.7118705558410667,1.324453906116211,-0.23682117358025612,-0.3468803446999865,0.0,0.0,
0.0,0.0027133711808662174,0.12135300439085207,-0.4922262363129788,-0.8598273777993808,0.5349750424696196,1.7689109486773893,0.8673056169876816,-0.12028057720927395,-0.5735353915331698,-0.7104958665942236,-1.144003722691284,-1.3667147175819794,-1.2051556979858304,1.5860121171540693,1.8277800894770748,-0.6556054804157967,2.279548014684691,3.2800742858633023,0.07813499282176445,-0.6490137883273963,-0.1767666727046058,0.0,0.0,
0.0,-0.007586954083802042,-0.19605947513968427,-0.32468981499512456,-0.7865897033917001,-1.490510421426915,-0.8989365517220734,-1.4302651282821606,-2.1024002175558265,-1.6801364390241196,-1.5406074086121844,-1.2722360304579396,-0.9835061570355017,1.2039323880082222,3.75753636608352,1.8780966167248936,2.4146592595845635,4.534405498524309,2.030019143131515,-0.34969727496592035,-0.4980620614081134,-0.015564477709501603,0.0,0.0,
0.0,0.0,-0.0025289846946006805,-0.09150648761146463,-0.3042143338340339,-0.48969792711528437,-0.8067652886364254,-0.6508543543649967,-0.23658006118442437,0.0,0.0,0.0,0.39376502480726844,3.363482301914057,4.085321585711715,1.1411723741169641,3.185284457898989,4.044599966909264,0.3903588056803493,-0.6258063295490994,-0.21473273599377732,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.015557913692854093,1.521292788905925,4.150931096301458,2.8554525406477196,1.7153798361438957,4.082346501773258,2.5743125474161133,-0.26780102174161075,-0.46225299392498004,-0.04504725100151718,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2279983397893424,3.0745042570944134,4.024364185586112,1.2300490054598412,3.059752918736021,4.216548577050863,0.9281672196789731,-0.5747620590750516,-0.33605329213211116,-0.009338686625700962,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.4604964902644655,4.341598628681333,2.77999970585073,1.4792678599598408,3.787705087427904,2.9707265271790644,0.03905621873030022,-0.5745420200628928,-0.08093528408940834,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4045057560142064,3.544031469623598,3.973098871820006,1.0415531509912392,2.860303714864478,3.907177300694119,1.0193794247518118,-0.4917623528279968,-0.40060912494742185,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.8828992743808384,4.143513216961609,2.4351140707904633,1.5838338219542802,3.8141541607822647,2.8322254700919944,-0.05094376126491919,-0.6591459263962036,-0.13843411725852678,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7001061161784342,3.762548120498499,3.897313592063377,1.0807050043510495,3.1384092844548115,3.986564034040211,0.8443761548552055,-0.5335610249741848,-0.40035094731074033,-0.009338686625700962,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.21781079169995732,2.6696918650516435,4.284368152330222,1.9789991695336742,1.8650058676222718,3.7817228689517663,2.2533401185069475,-0.16510265600941393,-0.6964346202781952,-0.11566603952194943,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.231679631751178,3.924062289138381,3.1035418990429093,0.8205578098121126,3.271341858707334,3.5744634436407985,0.6598786886475968,-0.608546228596166,-0.33637935536153535,-0.0031128955419003207,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4822953244784769,3.157606917512575,3.7639251274964667,0.8495410652780289,2.5006004843267964,3.824480560747324,1.73610165630129,-0.2514465856362916,-0.6770176631407476,-0.0747094930056077,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.17113705062139503,2.3877215788331907,3.998908363484317,1.3990913221396748,1.3306082975392923,3.109068476795561,2.5535622939490574,0.12050799214694545,-0.8854751121960998,-0.2436041924994252,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.10890539584997866,1.6299738423585266,3.9403389728426212,1.5723389702864303,0.5337858593483313,2.529035159959233,2.841002467457379,0.8505747691738799,-0.7566805758661026,-0.4644232990474139,-0.024903164335202566,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,1.0359034855826494,3.530669967909544,1.4362564298991674,0.002715544654265259,2.235214528591431,2.760049278331695,1.5253100289525174,-0.5226130061359128,-0.7101809570979761,-0.09649976179890994,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
};
    
    Shape in_shape = {28,28,1};
    Shape out_shape = {24,24,2};
    float* conv_out = (float*)malloc(24*24*16*sizeof(float));
    conv_forward(inp, in_shape, conv_out, kernel_weight, 5, 1, 16);
    CU_ASSERT_TRUE(compare_float_arr(conv_out, target_out, 24*24*16, EPSILON));
    // printVector(conv_out, 4,"");
    free(conv_out);
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
    CU_add_test(suite, "test_conv_forward2", test_conv_forward_2);
    CU_add_test(suite, "test_conv_backward", test_conv_backward);
    
    CU_add_test(suite, "test_conv_backward_1", test_conv_backward_1);

    // 运行所有测试用例
    CU_basic_set_mode(CU_BRM_VERBOSE); 
    CU_basic_run_tests();

    // 清理测试框架
    CU_cleanup_registry();

    return 0;
}
