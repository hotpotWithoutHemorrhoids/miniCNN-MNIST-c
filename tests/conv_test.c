#include<CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "test_util.h"

Shape conv_forward(float *inp, Shape in_shape,float* out, float* conv_weights, int kernel_size, int stride, int channel){
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
    for(int c=0;c<channel;c++){
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
                mementun_row[j] = mementun_row[j]*MOMENTUM + lr*grad_w_ij;
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

        for(int z=0;z<out_z;z++){
            float* full_conv_dloss_z = full_conv_dloss + z*new_row*new_col;
            float* d_loss_z = d_loss + z*out_h*out_w;
            // 第z个卷积核的权重
            float* conv_weights_z = conv_weights + z*inp_z*kernel_size*kernel_size;
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
                    for(int inp_c=0;inp_c<inp_z;inp_c++){
                        float* conv_weights_z_inp_c = conv_weights_z + inp_c*kernel_size*kernel_size;
                        float* d_inp_c = d_inp + inp_c*inp_h*inp_w;
                        for (int k = 0; k < kernel_size; k++){
                            for (int l = 0; l < kernel_size; l++){
                                d_inp_ij += full_conv_dloss_z[(i+k)*new_col + j+l]*
                                            conv_weights_z_inp_c[(kernel_size-k-1)*kernel_size+kernel_size-l-1];
                            }
                        }
                        d_inp_c[i*inp_w+j] += d_inp_ij;
                    }
/*                     for(int inp_c=0;inp_c<inp_size.z; inp_c++){
                        float* d_inp_c = d_inp + inp_c*inp_h*inp_w;
                        d_inp_c[i*inp_w+j] += d_inp_ij;
                    } */
                }
            }
        }
        free(full_conv_dloss);
    }

    // update weights
    for (int out_c = 0; out_c < out_z; out_c++){
        for(int c=0;c<channel;c++){
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
