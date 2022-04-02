#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    clock_t histogram_computation_start, histogram_computation_end;
    clock_t equalization_start, equalization_end;
    double total_histogram_computation;
    double total_equalization;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram_computation_start = clock();
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_computation_end = clock();
    total_histogram_computation = (double)(histogram_computation_end - histogram_computation_start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Total time to compute histogram %lf\n", total_histogram_computation);

    equalization_start = clock();
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
    equalization_end = clock();
    total_equalization = (double)(equalization_end - equalization_start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Total time to equalize image %lf\n", total_equalization);
    
    return result;
}
