/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define ACCURACY  	0.00005F
#define maximum_dimensions 32
//#define DEBUG
//#define EXIT_ON_ERROR

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, int imageW, int imageH, int filterR) {
  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;
      
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
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter, int imageW, int imageH, int filterR) {
  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

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

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, int imageW, int imageH, int filterR){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;

  if(y > imageH - 1 || x > imageW - 1)
    return;

  for (k = -filterR; k <= filterR; k++) {
    int d = x + k;
    if(d >= 0 && d < imageW) {
      sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
    }     
    d_Dst[y * imageW + x] = sum;
  }
}

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter, int imageW, int imageH, int filterR) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;

  if(y > imageH - 1 || x > imageW -1)
    return;

  

  for (k = -filterR; k <= filterR; k++) {
    int d = y + k;

    if(d >= 0 && d < imageH) {
      sum += d_Src[d * imageW + x] *d_Filter[filterR - k];
    }   
    d_Dst[y * imageW + x] = sum;
  }   
}

void grid_init(dim3 *grid_dimensions, dim3 *block_dimensions, int imageW, int imageH){
  //Default Grid Values
  grid_dimensions->x = 1; 
  grid_dimensions->y = 1; 
  
  //Block Dimensions (0 - 32)
  block_dimensions->x = imageW > maximum_dimensions ? maximum_dimensions : imageW;
  block_dimensions->y = imageH > maximum_dimensions ? maximum_dimensions : imageH;


  //Calculate Grid's Horizontal Geometry
  if(imageW > maximum_dimensions){
    grid_dimensions->x = (imageW / maximum_dimensions); 
    if(imageW % maximum_dimensions != 0)
      grid_dimensions->x++;
  }
  
  //Calculate Grid's Vertical Geometry
  if(imageH > maximum_dimensions){
    grid_dimensions->y = (imageH / maximum_dimensions); 
    if(imageH % maximum_dimensions != 0)
      grid_dimensions->y++;
  }
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  double
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
  dim3 grid_dimensions; 
  int imageW;
  int imageH;
  double accuracy = ACCURACY;
  unsigned int i;
  clock_t h_start, h_end, d_start, d_end;
  double h_time_used, d_memcpy_time_used, d_computation_time_used, d_time_used;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  imageH = imageW;


  //printf("\nImage Width x Height = %i x %i\n", imageW, imageH);
  //printf("Filter lenght: %d, Filter radious: %d\n", FILTER_LENGTH, filter_radius);

  //printf("Allocating and initializing host arrays...\n");
  // Allocate memory on host
  h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
  h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
  h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
  h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
  h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));

  //Check malloc() for host
  if(!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU)
    return -1;

  srand(200);

  for (i = 0; i < FILTER_LENGTH; i++) {
      h_Filter[i] = (double)(rand() % 16);
  }

  for (i = 0; i < imageW * imageH; i++) {
      h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
  }

  // Start CPU computation
  printf("\033[0;31mCPU computation...\033[0m\n");

  //Start calculate CPU time
  h_start = clock();
 
  convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
 
  //End calculate CPU time
  h_end = clock();
  h_time_used = ((double) (h_end - h_start)) / CLOCKS_PER_SEC;

#ifdef DEBUG
  for(int i = 0; i < imageH; i++){
    for(int j = 0; j < imageW; j++){
      printf("%lf ", h_OutputCPU[i+j]);
    }
    printf("\n");
  }
#endif

  //printf("Allocating and initializing device arrays...\n");
  // Allocate memory on device
  cudaMalloc((void **) &d_Filter, FILTER_LENGTH * sizeof(double));
  cudaMalloc((void **) &d_Input, imageW * imageH * sizeof(double));
  cudaMalloc((void **) &d_Buffer, imageW * imageH * sizeof(double));
  cudaMalloc((void **) &d_OutputGPU, imageW * imageH * sizeof(double));

  //Check malloc() for device
  if(!d_Filter || !d_Input || !d_Buffer || !d_OutputGPU)
    return -1;

 
  //Initialize Grid
  grid_init(&grid_dimensions, &block_dimensions, imageW, imageH);

  printf("Grid(%d, %d), Block(%d, %d)\n", grid_dimensions.x, grid_dimensions.y, block_dimensions.x, block_dimensions.y);
  
  // Start GPU computation
  printf("\033[0;32mGPU computation...\033[0m\n");
  //Start calculate GPU memory copy time
  d_start = clock();

  //Copying data from host to device
  cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(double), cudaMemcpyHostToDevice);
  
  d_end = clock();
  d_memcpy_time_used = ((double) (d_end - d_start)) / CLOCKS_PER_SEC;


  d_start = clock();
  //Kernel call
  convolutionRowGPU<<<grid_dimensions, block_dimensions>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius); //First Kernel must finish copying data
  convolutionColumnGPU<<<grid_dimensions, block_dimensions>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius); //After copying is finished launch
  
  //End calculate GPU time
  d_end = clock();
  d_computation_time_used = ((double) (d_end - d_start)) / CLOCKS_PER_SEC;


   d_start = clock();
  //Copying data from device to host
  cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(double), cudaMemcpyDeviceToHost);
  d_end = clock();
  d_memcpy_time_used += ((double) (d_end - d_start)) / CLOCKS_PER_SEC;
  
  
  //printf("%s: %s\n", cudaGetErrorName(cudaGetLastError()), cudaGetErrorString(cudaGetLastError()));
  
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
    if(ABS(h_OutputCPU[i] - h_OutputGPU[i]) > accuracy){
      accuracy = ABS(h_OutputCPU[i] - h_OutputGPU[i]);
#ifdef EXIT_ON_ERROR
     printf("CPU and GPU Produce Different Results at [%d][%d]: %lf\n", i / imageH, i % imageW, accuracy);  
     break;
#endif
    }
  }

  printf("Comparing CPU and GPU with Accuracy %lf\n", accuracy);

  printf("\033[0;31mCPU Computation = %lf\033[0m\n", h_time_used);
  printf("\033[0;32mGPU Time = Memory Copy(%lf) + Computation(%lf) = %lf\033[0m\n", d_memcpy_time_used, d_computation_time_used, d_memcpy_time_used + d_computation_time_used);
  
  
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
