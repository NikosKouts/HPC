/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define ACCURACY  	0.00005

//#define DEBUG

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, int imageW, int imageH, int filterR) {
  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;
      
      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        if(d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter, int imageW, int imageH, int filterR) {
  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if(d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }   
}

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, int imageW, int imageH, int filterR){
  int x = threadIdx.x;
  int y = threadIdx.y;
  float sum = 0;
  int k;
  
  for (k = -filterR; k <= filterR; k++) {
    int d = x + k;
    if(d >= 0 && d < imageW) {
      sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
    }     

    d_Dst[y * imageW + x] = sum;
  }
}

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter, int imageW, int imageH, int filterR) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  float sum = 0;
  int k;

  for (k = -filterR; k <= filterR; k++) {
    int d = y + k;

    if(d >= 0 && d < imageH) {
      sum += d_Src[d * imageW + x] *d_Filter[filterR - k];
    }   

    d_Dst[y * imageW + x] = sum;
  }   
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  float
  *h_Filter,
  *h_Input,
  *h_Buffer,
  *h_OutputCPU, 
  *h_OutputGPU,
  *d_Filter,
  *d_Input,
  *d_Buffer,
  *d_OutputGPU;
  dim3 block_dimensions;
  int imageW;
  int imageH;
  double accuracy = ACCURACY;
  unsigned int i;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  block_dimensions.x = block_dimensions.y = imageH = imageW;
  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);


  printf("Allocating and initializing host arrays...\n");
  // Allocate memory on host
  h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
  h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
  h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));

  //Check malloc() for host
  if(!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU)
    return -1;

  srand(200);

  for (i = 0; i < FILTER_LENGTH; i++) {
      h_Filter[i] = (float)(rand() % 16);
  }

  for (i = 0; i < imageW * imageH; i++) {
      h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
  }


  // Start CPU computation
  printf("\033[0;31mCPU computation...\n\033[0m");
  convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

#ifdef DEBUG
  for(int i = 0; i < imageH; i++){
    for(int j = 0; j < imageW; j++){
      printf("%lf ", h_OutputCPU[i+j]);
    }
    printf("\n");
  }
#endif

  printf("Allocating and initializing device arrays...\n");
  // Allocate memory on device
  cudaMalloc((void **) &d_Filter, FILTER_LENGTH * sizeof(float));
  cudaMalloc((void **) &d_Input, imageW * imageH * sizeof(float));
  cudaMalloc((void **) &d_Buffer, imageW * imageH * sizeof(float));
  cudaMalloc((void **) &d_OutputGPU, imageW * imageH * sizeof(float));

  //Check malloc() for device
  if(!d_Filter || !d_Input || !d_Buffer || !d_OutputGPU)
    return -1;
  
  //Copying data from host to device
  cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);

  // Start GPU computation
  printf("\033[0;32mGPU computation...\033[0m\n");
  
  //Kernel call
  convolutionRowGPU<<<1, block_dimensions>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius); //First Kernel must finish copying data
  convolutionColumnGPU<<<1, block_dimensions>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius); //After copying is finished launch

  //Copying data from device to host
  cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);
  
  fprintf(stderr, "%s: %s\n", cudaGetErrorName(cudaGetLastError()), cudaGetErrorString(cudaGetLastError()));

#ifdef DEBUG
  for(int i = 0; i < imageH; i++){
    for(int j = 0; j < imageW; j++){
      printf("%lf ", h_OutputGPU[i+j]);
    }
    printf("\n");
  }
#endif

  // Compare CPU and GPU results
  for(int i = 0; i < imageH * imageW; i++){
    if(ABS(h_OutputCPU[i] - h_OutputGPU[i]) > ACCURACY){
      accuracy = ABS(h_OutputCPU[i] - h_OutputGPU[i]);
     //fprintf(stderr, "CPU and GPU Produce Different Results at [%d][%d]: %lf\n", i / imageH, i % imageW, ABS(h_OutputCPU[i] - h_OutputGPU[i]));  
     //break;
    }
  }
  fprintf(stderr, "\033[0mComparing CPU and GPU with Accuracy %lf\n", accuracy);



  // Free all the allocated host memory ja
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Filter);

  // Free all the allocated device memory
  cudaFree(d_Filter);
  cudaFree(d_Input);
  cudaFree(d_Buffer);
  cudaFree(d_OutputGPU);

  // Do a device reset just in case...
  cudaDeviceReset();
  
  return 0;
}
