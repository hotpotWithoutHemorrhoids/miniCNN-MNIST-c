#ifndef DATALOADER_H
#define DATALOADER_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include"utils.h"

typedef struct{

    size_t row;
    size_t col;
}PictureSize;


typedef struct{
    float* data;
    int* labels;
}Data;

typedef struct {
    size_t nImages;
    size_t nLabels;
    PictureSize imageSize;

    FILE* imagesfile;
    FILE* labelsfile;

    int should_shuffer;
    unsigned char* images;
    unsigned char* labels;
    size_t header_bytes;
    size_t image_bytes;
} DataLoader;

void shuffle_data(DataLoader* dataloader){
    if(dataloader->should_shuffer){
        for(int i=dataloader->nImages-1; i>1; i--){
            int j = rand() % i;
            unsigned char* i_images = dataloader->images + i*dataloader->imageSize.row*dataloader->imageSize.col;
            unsigned char* j_images = dataloader->images + j*dataloader->imageSize.row*dataloader->imageSize.col;
            for(int k=0;k<dataloader->imageSize.row*dataloader->imageSize.col;k++){
                unsigned char temp = i_images[k];
                i_images[k] = j_images[k];
                j_images[k] = temp;
            }
            int label_i = dataloader->labels[i];
            dataloader->labels[i] = dataloader->labels[j];
            dataloader->labels[j] = label_i;
        }
    }
}

void dataloader_init(DataLoader* dataloader, const char* images_path, const char* labels_path, int should_shuffer){
    dataloader->imagesfile = fopenCheck(images_path, "rb");
    dataloader->labelsfile = fopenCheck(labels_path, "rb");
    dataloader->labels = NULL;
    dataloader->images = NULL;

    dataloader->should_shuffer = should_shuffer;
    int temp;
    freadCheck(&temp, sizeof(int), 1, dataloader->imagesfile);
    freadCheck(&dataloader->nImages, sizeof(int), 1, dataloader->imagesfile);
    freadCheck(&dataloader->imageSize.row, sizeof(int), 1, dataloader->imagesfile);
    freadCheck(&dataloader->imageSize.col, sizeof(int), 1, dataloader->imagesfile);

    dataloader->nImages = __builtin_bswap32(dataloader->nImages);
    dataloader->imageSize.row = __builtin_bswap32(dataloader->imageSize.row);
    dataloader->imageSize.col = __builtin_bswap32(dataloader->imageSize.col);

    printf("read %s  temp: %d, nImages: %d, rows: %d, cols: %d\n",images_path,\
         temp, (int)dataloader->nImages, (int)dataloader->imageSize.row, (int)dataloader->imageSize.col);

    freadCheck(&temp, sizeof(int), 1, dataloader->labelsfile);
    freadCheck(&dataloader->nLabels, sizeof(int), 1, dataloader->labelsfile);

    dataloader->nLabels = __builtin_bswap32(dataloader->nLabels);

    printf("%s temp: %d, nLabels: %d\n",labels_path, temp, (int)dataloader->nLabels);
    // dataloader->images = (float*)malloc(dataloader->nImages * dataloader->imageSize.row * dataloader->imageSize.col * sizeof(float));

    dataloader->images = (unsigned char*)malloc(dataloader->nImages * dataloader->imageSize.row * dataloader->imageSize.col * sizeof(unsigned char));
    freadCheck(dataloader->images, sizeof(unsigned char), dataloader->nImages * dataloader->imageSize.row * dataloader->imageSize.col, dataloader->imagesfile);

    dataloader->labels = (unsigned char*)malloc(dataloader->nLabels * sizeof(unsigned char));
    freadCheck(dataloader->labels, sizeof(unsigned char), dataloader->nLabels, dataloader->labelsfile);

    if (should_shuffer){
        shuffle_data(dataloader);
    }

    fcloseCheck(dataloader->imagesfile);
    fcloseCheck(dataloader->labelsfile);
}

void DataLoader_clear(DataLoader *dataloader){
    if(dataloader->images != NULL)
        free(dataloader->images);
    if(dataloader->labels != NULL)
        free(dataloader->labels);
}



void load_betch_images(DataLoader* dataloader, Data* datas,int idx, int batch){
    unsigned char* images = dataloader->images + idx*batch*dataloader->imageSize.row*dataloader->imageSize.col;
    // for(int i=0;i<batch*dataloader->imageSize.row*dataloader->imageSize.col; i++){
    //     dst[i] = (float)images[i];
    // }
    int row = dataloader->imageSize.row;
    int col = dataloader->imageSize.col;

    for(int i=0;i<batch;i++){
        datas->labels[i] = dataloader->labels[idx*batch+i];
        for(int j=0;j<row*col;j++){
            // 归一化
            // datas->data[i*row*col+j] = ((float)images[i*row*col+j])/255.0f;
            datas->data[i*row*col+j] = ((float)images[i*row*col+j]);

        }
    }
}


#endif

