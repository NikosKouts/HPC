#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    cudaMallocHost(&result.img, result.w * result.h * sizeof(unsigned char));
    cudaCheckError();

    
    gpu_equalization(hist, img_in.img, result.img, img_in.h * img_in.w, 256);
    //histogram_equalization_gpu(tiles, result.img,img_in.img,hist,result.w*result.h, 256);
    return result;
}
