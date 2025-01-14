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
#include <float.h>


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
                float max = -FLT_MAX;
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
                float max = -FLT_MAX;
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
    float inp[] = {
        -0.9414,  1.2632, -0.1838,  0.1505,  0.1075, -0.2780, -2.6021,  0.6245,
        -0.8684, -0.2051,  0.3976,  0.6699, -0.0537,  0.0467, -1.7671, -2.1205,
         1.5191, -0.6682,  0.0031, -0.1535, -0.2345,  1.6892,  0.2716, -0.1365,
        -0.7042,  2.0126,  0.9120,  0.2773,  0.8201, -0.9151, -2.1437,  1.4072,
         1.2744, -0.1874,  2.1762, -0.5738
    };

    float target_out[] ={
        1.2632, 0.1505, 0.6699, 0.0467, 1.6892, 1.5191, 2.0126, 1.2744, 2.1762
    };
    int kernelsize = 2;

    float out[9];

    pool_froward(
        inp, 6,6,1,out,kernelsize
    );

    CU_ASSERT_TRUE(compare_float_arr(out, target_out, 9, EPSILON));
}


void test_pool_forward1(){
    float inp[] = {
        -0.9414,  1.2632, -0.1838,  0.1505,  0.1075, -0.2780, -2.6021,  0.6245,
        -0.8684, -0.2051,  0.3976,  0.6699, -0.0537,  0.0467, -1.7671, -2.1205,
         1.5191, -0.6682,  0.0031, -0.1535,  1.1396, -0.2302,  1.1877,  0.7677,
        -0.7588, -0.1853, -0.8558, -0.2346, -0.4215,  0.8488, -0.6776, -0.9445,
        -0.4815,  1.2434,  2.3693,  0.2829, -0.2345,  1.6892,  0.2716, -0.1365,
        -0.6948, -1.3186, -0.9694,  0.6403,  0.8201, -0.9151, -2.1437,  1.4072,
        -0.0263,  2.7204, -0.5955,  0.9871,  1.0861,  0.0610,  0.0417,  0.6783,
        -0.8952, -1.0143, -0.2429, -1.5727,  1.3940, -0.1941,  0.0048, -1.3165,
         0.0204, -0.1652,  0.2109,  0.5167,  0.2430, -0.3522, -0.7664,  0.2973,
         1.2710,  1.8117,  1.5320, -0.2602, -1.1611, -1.2380,  1.0533,  0.2002,
         0.8206,  2.5164, -0.9537, -0.0167,  0.2526,  0.6762,  1.7228, -0.4043,
         1.0915,  0.4550, -1.2863, -0.4498, -1.0524, -0.4481, -0.5325, -0.1469,
        -0.3918,  0.1269, -0.8301, -1.6481, -0.1881,  0.8706, -1.4714, -1.1864,
        -0.2659,  1.0877, -0.9446, -0.7943
    };

    float target_out[] ={
       1.2632,  0.1505,  0.6699,  0.0467,  1.1396,  1.5191, -0.1853,  1.2434,
         2.3693,  1.6892,  0.8201,  1.4072,  2.7204,  0.9871,  1.0861,  1.3940,
         0.2430,  0.2973,  1.8117,  2.5164, -0.0167,  0.6762,  1.7228,  1.0915,
         0.1269,  1.0877,  0.8706
    };
    int kernelsize = 2;

    float out[27];

    pool_froward(
        inp, 6,6,3,out,kernelsize
    );

    CU_ASSERT_TRUE(compare_float_arr(out, target_out, 27, EPSILON));
    
}


void test_pool_backward(){

    Shape in_shape = {6,6,3};
    float inp[] = {-1.5256, -0.7502, -0.6540, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091,
        -0.7121,  0.3037, -0.7773, -0.2515, -0.2223,  1.6871,  0.2284,  0.4676,
        -0.6970, -1.1608,  0.6995,  0.1991,  0.8657,  0.2444, -0.6629,  0.8073,
         1.1017, -0.1759, -2.2456, -1.4465,  0.0612, -0.6177, -0.7981, -0.1316,
         1.8793, -0.0721,  0.1578, -0.7735,  0.1991,  0.0457,  0.1530, -0.4757,
        -0.1110,  0.2927, -0.1578, -0.0288,  2.3571, -1.0373,  1.5748, -0.6298,
        -0.9274,  0.5451,  0.0663, -0.4370,  0.7626,  0.4415,  1.1651,  2.0154,
         0.1374,  0.9386, -0.1860, -0.6446,  1.5392, -0.8696, -3.3312, -0.7479,
        -0.0255, -1.0233, -0.5962, -1.0055, -0.2106, -0.0075,  1.6734,  0.0103,
        -0.7040, -0.1853, -0.9962, -0.8313, -0.4610, -0.5601,  0.3956, -0.9823,
        -0.5065,  0.0998, -0.6540,  0.7317, -1.4344, -0.5008,  0.1716, -0.1600,
         0.2546, -0.5020, -1.0412,  0.7323,  1.2466,  0.5057,  0.9505,  1.2966,
         0.8738, -0.5603,  1.2858,  0.8168, -1.4648, -1.2629,  1.1220,  1.5663,
         2.5581, -0.2334, -0.0135,  1.8606};

    Shape out_shape = {3,3,3};
    float d_out[] = {-1.6588,  0.1705,  0.0135, -0.3455,  0.6124,  1.8138, -0.9916, -2.0142,
         0.0228,  1.0375,  0.0154,  1.3161,  3.2301,  0.9076, -0.4015, -0.8243,
        -0.6287, -0.6191, -2.1734, -1.2728,  2.7071,  1.0864, -0.4109, -0.9835,
         1.6947,  1.5457, -1.0215};

    float target_dinp[] = {0.0000, -1.6588,  0.0000,  0.0000,  0.0135,  0.0000,  0.0000,  0.0000,
         0.0000,  0.1705,  0.0000,  0.0000,  0.0000, -0.3455,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.6124,  0.0000,  0.0000,  1.8138,
        -0.9916,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        -2.0142,  0.0000,  0.0228,  0.0000,  1.0375,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0154,  0.0000,  1.3161,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000, -0.4015,  0.0000,  0.0000,  3.2301,
         0.0000,  0.9076,  0.0000,  0.0000, -0.8243,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.6287, -0.6191,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -2.1734,  0.0000,
         0.0000, -1.2728,  0.0000,  2.7071,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  1.0864, -0.4109,  0.0000,  0.0000, -0.9835,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.6947,
         1.5457,  0.0000,  0.0000, -1.0215};
    
    float d_inp[6*6*3];
    int pool_size=2;
    printVector(target_dinp, 6*6*3, "pool backward target_dinp");
    pool_backward(inp, in_shape, d_out, out_shape, d_inp, pool_size);
    
    CU_ASSERT_TRUE(compare_float_arr(d_inp, target_dinp, 6*6*3, EPSILON));
    printVector(d_inp, 6*6*3, "pool backward d_inp");
}

// 初始化测试套件
int init_suite(void) {
    return 0;
}

// 清理测试套件
int clean_suite(void) {
    return 0;
}


// 编译：  gcc pool_test.c -o pool_test -I../include -lcunit
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
    CU_add_test(suite, "test_pool_forward1", test_pool_forward1);

    CU_add_test(suite, "test_pool_backward", test_pool_backward);
    

    // 运行所有测试用例
    CU_basic_set_mode(CU_BRM_VERBOSE); 
    CU_basic_run_tests();

    // 清理测试框架
    CU_cleanup_registry();

    return 0;
}

