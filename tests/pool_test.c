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

    #ifdef LOG
    for(int i=0;i<z; i++){
        char buff[50];
        snprintf(buff, sizeof(buff), "pool_froward result, channel %d", i);
        printMatrix(out, out_h, out_w, buff);
    }
    printf("===================================\n");
    #endif
    return output_shape;
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
                float max = 0.0f;
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

#define EPSILON 0.00001
void test_pool_forward(){

}

void test_pool_backward(){

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

    CU_pSuite suite = CU_add_suite("pool layer test", init_suite, clean_suite);
    if(suite == NULL){
        CU_cleanup_registry();
        return CU_get_error();
    }

    // 添加测试用例到测试套件中
    CU_add_test(suite, "test_pool_forward", test_pool_forward);
    CU_add_test(suite, "test_pool_backward", test_pool_backward);

    // 运行所有测试用例
    CU_basic_set_mode(CU_BRM_VERBOSE); 
    CU_basic_run_tests();

    // 清理测试框架
    CU_cleanup_registry();

    return 0;
}

