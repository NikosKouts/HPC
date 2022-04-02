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
#define ACCURACY  	0.00005
#define maximum_dimensions 32
//#define DEBUG

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

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, int padded_imageW, int imageH, int filterR, int padding_start_x, int padding_start_y){
  int x = blockIdx.x * blockDim.x + threadIdx.x + padding_start_x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;
  int position = y * padded_imageW + padding_start_y + x;

  for (k = -filterR; k <= filterR; k++) {
    int d = x + k;
    
    sum += d_Src[y * padded_imageW + d + padding_start_y] * d_Filter[filterR - k];
  
    d_Dst[position] = sum;
  }
}

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter, int padded_imageW, int imageH, int filterR, int padding_start_x, int padding_start_y) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + padding_start_x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;
  int position = y * padded_imageW + padding_start_y + x;

  for (k = -filterR; k <= filterR; k++) {
    int d = y + k;

    sum += d_Src[d * padded_imageW + padding_start_y + x] * d_Filter[filterR - k];
       
    d_Dst[position] = sum;
  }   
}

void grid_init(dim3 *grid_dimensions, dim3 *block_dimensions, int imageW, int imageH, int *padded_imageW, int *padded_imageH){
  //Block Dimensions (0 - 32)
  block_dimensions->x = imageW > maximum_dimensions ? maximum_dimensions : imageW;
  block_dimensions->y = imageH > maximum_dimensions ? maximum_dimensions : imageH;

  //Default Grid Values
  grid_dimensions->x = 1; 
  grid_dimensions->y = 1; 

  //Default Array size
  *padded_imageW = *padded_imageH = imageW + 2 * filter_radius;
  
  //Calculate new grid dimensions and the size of the new array
  if(imageW > maximum_dimensions){
    grid_dimensions->x = (imageW / maximum_dimensions); 
    if(imageW % maximum_dimensions != 0){
        grid_dimensions->x++;

        //Calculate new width of the array
        if(grid_dimensions->x * maximum_dimensions > imageW + filter_radius)
          *padded_imageW = grid_dimensions->x * maximum_dimensions + 2 * filter_radius;
    }
  }

  if(imageH > maximum_dimensions){
    grid_dimensions->y = (imageH / maximum_dimensions); 
    if(imageH % maximum_dimensions != 0) {
      grid_dimensions->y++;

      //Calculate new height of the array
      if(grid_dimensions->y * maximum_dimensions > imageH + filter_radius){
        *padded_imageH = grid_dimensions->y * maximum_dimensions + 2 * filter_radius;
      }
    }
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
  *d_OutputGPU,
  *padding_Input;
  dim3 block_dimensions;
  dim3 grid_dimensions; 
  int imageW;
  int imageH;
  int padded_imageW, padded_imageH, position;
  int padding_start_x, padding_start_y;
  double accuracy = ACCURACY;
  unsigned int i;
  clock_t h_start, h_end, d_start, d_end;
  double h_time_used, d_memcpy_time_used, d_computation_time_used, d_time_used;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);
  
  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  imageH = imageW;

  // Allocate memory on host
  h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
  h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
  h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
  h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
  //Check malloc() for host
  if(!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU)
    return -1;
    
  srand(200);

  for (i = 0; i < FILTER_LENGTH; i++) {
      h_Filter[i] = (double)(rand() % 16);
  }

  for (i = 0; i < imageW * imageH; i++) {
      h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
  }

  // Start CPU computation
  printf("\033[0;31mCPU computation...\n\033[0m");

  //Start calculate cpu time
  h_start = clock();
 
  convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
  
  //End calculate CPU time
  h_end = clock();
  h_time_used = ((double) (h_end - h_start)) / CLOCKS_PER_SEC;

#ifdef DEBUG
  for(int i = 0; i < imageH; i++){
    for(int j = 0; j < imageW; j++){
      printf("%lf ", h_OutputCPU[i*imageW+j]);
    }
    printf("\n");
  }
#endif


  grid_init(&grid_dimensions, &block_dimensions, imageW, imageH, &padded_imageW, &padded_imageH);

  printf("Grid(%d, %d), Block(%d, %d)\n", grid_dimensions.x, grid_dimensions.y, block_dimensions.x, block_dimensions.y);

  // Start GPU computation
  printf("\033[0;32mGPU computation...\033[0m\n");

  // Allocate memory on device
  cudaMalloc((void **) &d_Filter, FILTER_LENGTH * sizeof(double));
  cudaMalloc((void **) &d_Input, padded_imageW * padded_imageH * sizeof(double));
  cudaMalloc((void **) &d_Buffer, padded_imageW * padded_imageH * sizeof(double));
  cudaMalloc((void **) &d_OutputGPU, padded_imageW * padded_imageH * sizeof(double));
  padding_Input = (double *) calloc(padded_imageW * padded_imageH, sizeof(double));
  h_OutputGPU = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));

  //Check malloc() for device
  if(!d_Filter || !d_Input || !d_Buffer || !d_OutputGPU || !padding_Input || !h_OutputGPU)
    return -1;

  //Padding
  position = 0;
  padding_start_y = padded_imageW * filter_radius;
  padding_start_x = filter_radius;

  for(int i = 0; i < imageH; i++) {
    for(int j = 0; j < imageW; j++){
      padding_Input[i * padded_imageW + padding_start_y + j + padding_start_x] = h_Input[position];
      position++;
    }
  }

  //start calculate GPU time
  d_start = clock();

  //Copying data from host to device
  cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Input, padding_Input, padded_imageW * padded_imageH * sizeof(double), cudaMemcpyHostToDevice);

  d_end = clock();
  d_memcpy_time_used = ((double) (d_end - d_start)) / CLOCKS_PER_SEC;

  d_start = clock();
  //Kernel call
  convolutionRowGPU<<<grid_dimensions, block_dimensions>>>(d_Buffer, d_Input, d_Filter, padded_imageW, imageH, filter_radius, padding_start_x, padding_start_y); //First Kernel must finish copying data
  convolutionColumnGPU<<<grid_dimensions, block_dimensions>>>(d_OutputGPU, d_Buffer, d_Filter, padded_imageW, imageH, filter_radius, padding_start_x, padding_start_y); //After copying is finished launch

  //End calculate GPU time  
  d_end = clock();
  d_computation_time_used = ((double) (d_end - d_start)) / CLOCKS_PER_SEC;


  d_start = clock();
  //Copying data from device to host
  cudaMemcpy(h_OutputGPU, d_OutputGPU, padded_imageW * padded_imageW * sizeof(double), cudaMemcpyDeviceToHost);

  //End calculate GPU time
  d_end = clock();
  d_memcpy_time_used += ((double) (d_end - d_start)) / CLOCKS_PER_SEC;
  
  fprintf(stderr, "%s: %s\n", cudaGetErrorName(cudaGetLastError()), cudaGetErrorString(cudaGetLastError()));
  
#ifdef DEBUG
  for(int i = 0; i < imageH; i++){
    for(int j = 0; j < imageW; j++){
      printf("%lf ", h_OutputGPU[i*imageW+j]);
    }
    printf("\n");
  }
#endif

  // Compare CPU and GPU results
  for(int i = 0; i < imageH; i++){
    for(int j = 0; j < imageW; j++) {
      if(ABS(h_OutputCPU[i*imageW + j] - h_OutputGPU[i*padded_imageW + padding_start_y + j + padding_start_x]) > ACCURACY){
        accuracy = ABS(h_OutputCPU[i*imageW + j] - h_OutputGPU[i*padded_imageW + padding_start_y + j + padding_start_x]);   
#ifdef EXIT_ON_ERROR
      printf("CPU and GPU Produce Different Results: %lf\n",accuracy);  
      break;
#endif
      }
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
  free(padding_Input);

  // Free all the allocated device memory
  cudaFree(d_Filter);
  cudaFree(d_Input);
  cudaFree(d_Buffer);
  cudaFree(d_OutputGPU);

  // Do a device reset just in case...
  cudaDeviceReset();
  
  return 0;
}
