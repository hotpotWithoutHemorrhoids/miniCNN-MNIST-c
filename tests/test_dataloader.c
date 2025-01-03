

/* 
test about dataloader.h with Cunit
*/

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataloader.h"

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

void test_dataloader_init(){
    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);
    CU_ASSERT_EQUAL(dataloader.nImages, 60000);
    CU_ASSERT_EQUAL(dataloader.nLabels, 60000);
    CU_ASSERT_EQUAL(dataloader.imageSize.row, 28);
    CU_ASSERT_EQUAL(dataloader.imageSize.col, 28);

    CU_ASSERT_PTR_NOT_NULL(dataloader.images);
    CU_ASSERT_PTR_NOT_NULL(dataloader.labels);

    CU_ASSERT_EQUAL(dataloader.should_shuffer, 1);
}

// test load_betch_images
void test_load_betch_images(){
    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);
    Data datas;
    const int BATCH = 1000;
    int row = dataloader.imageSize.row, col = dataloader.imageSize.col;
    datas.data = (float*)malloc(BATCH*row*col*sizeof(float));
    datas.labels = (int*)malloc(BATCH*sizeof(int));
    load_betch_images(&dataloader, &datas, 0, BATCH);

    int t = rand() % BATCH;

    float* t_image = datas.data + t*row*col;
    int label = datas.labels[t];
    printf("label: %d, row: %d, col: %d\n", label, row, col);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            printf("%3.0f ", t_image[i*col+j]);
        }
        printf("\n");
    }
    printf("\n");
    free(datas.data);
    free(datas.labels);
}


void test_load_betch_images_1(){
    DataLoader dataloader;
    dataloader_init(&dataloader, TRAIN_IMG_PATH, TRAIN_LBL_PATH, 1);
    Data datas;
    const int BATCH = 1000;
    int row = dataloader.imageSize.row, col = dataloader.imageSize.col;
    datas.data = (float*)malloc(BATCH*row*col*sizeof(float));
    datas.labels = (int*)malloc(BATCH*sizeof(int));
    
    int train_size = (int)(dataloader.nImages* 0.8);
    for(int b=0;b<train_size/BATCH;b++){
        load_betch_images(&dataloader, &datas, b, BATCH);
    }

    int t = rand() % BATCH;

    float* t_image = datas.data + t*row*col;
    int label = datas.labels[t];
    printf("label: %d, row: %d, col: %d\n", label, row, col);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            printf("%3.0f ", t_image[i*col+j]);
        }
        printf("\n");
    }
    printf("\n");
    free(datas.data);
    free(datas.labels);
}


// 如何编译本文件: gcc -o test_dataloader tests/test_dataloader.c src/dataloader.h src/utils.h -Iinclude -lcunit

int main(int argc, char const *argv[])
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("dataloader_test", 0, 0);
    CU_add_test(suite, "test_dataloader_init", test_dataloader_init);
    CU_add_test(suite, "test_load_betch_images", test_load_betch_images);
    CU_add_test(suite, "test_load_betch_images_1", test_load_betch_images_1);
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}


