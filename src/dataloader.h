#ifndef DATALOADER_H
#include<stdio.h>
#include<stdlib.h>
#include<utils.h>


typedef struct{

    size_t row;
    size_t col;
}PictureSize;


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

void dataloader_init(DataLoader* dataloader, const char* images_path, const char* labels_path, int should_shuffer){
    dataloader->imagesfile = fopenCheck(images_path, "rb");
    dataloader->labelsfile = fopenCheck(labels_path, "rb");

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


    fcloseCheck(dataloader->imagesfile);
    fcloseCheck(dataloader->labelsfile);
}


#endif
