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
#define BLOCK_DIMENSION 32
//#define CPU

#define cudaCheckError() {                                                                        \
  cudaError_t error=cudaGetLastError();                                                           \
  if(error!=cudaSuccess) {                                                                        \
    fprintf(stderr, "ERROR IN CUDA %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));   \
    cudaDeviceReset();                                                                            \
    exit(EXIT_FAILURE);                                                                           \
  }                                                                                               \
}



// Constant Memory
__constant__ double c_Filter[8192]; //Maximum Constant Memory


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

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, int imageW, int imageH, int filterR){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;
  int d;
  int left_side, right_side;

  // Padded Shared Memory
  extern __shared__ double s_src[];
  
  // Filter Left Side
  left_side = x - filterR;

  // Leftmost Block 
  if(left_side < 0){
    s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx] = 0;
  }
  // Blocks Located Right or Middle of the Grid 
  else if(tx < filterR){
    s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx] = d_Src[y * imageW + x - filterR];
  }

  // Filter Right Side
  right_side = x + filterR;
  
  // Rightmost Blocks
  if(right_side > imageW - 1){
    s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx + 2 * filterR] = 0;
  }
  // Blocks Located in the Left or Middle  of the Grid
  else if(BLOCK_DIMENSION - 1 - tx < filterR){
    s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx + 2 * filterR] = d_Src[y * imageW + x + filterR];
  }


  //Actual Value
  s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx + filterR] = d_Src[y * imageW + x];

  __syncthreads();



  //Actual Convolution
  for (k = -filterR; k <= filterR; k++) {
    d = tx + k;

    sum += s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + d + filterR] * c_Filter[filterR - k];  
  }
  
  d_Dst[y * imageW + x] = sum;
}

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, int imageW, int imageH, int filterR) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;
  int d;
  int top_side, bottom_side;

  // Padded Shared Memory
  extern __shared__ double s_src[];
  
  // Filter Top Side
  top_side = y - filterR;

  // Uppermost Block 
  if(top_side < 0){
    s_src[ty * BLOCK_DIMENSION + tx] = 0;
  }
  else if(ty < filterR){
    s_src[ty * BLOCK_DIMENSION + tx] = d_Src[(y - filterR) * imageW + x];
  }

  // Filter Bottom Side
  bottom_side = y + filterR;
  
  // Bottommost Blocks
  if(bottom_side > imageH - 1){
    s_src[(ty + 2 * filterR) * BLOCK_DIMENSION + tx] = 0;
  }
  else if(BLOCK_DIMENSION - 1 - ty < filterR){
    s_src[(ty + 2 * filterR) * BLOCK_DIMENSION + tx] = d_Src[(y + filterR) * imageW + x];
  }
  
  s_src[(ty + filterR) * BLOCK_DIMENSION + tx] = d_Src[y * imageW + x];

  __syncthreads();


  for (k = -filterR; k <= filterR; k++) {
    d = ty + k;

    sum += s_src[(d + filterR) * BLOCK_DIMENSION + tx] * c_Filter[filterR - k];
  }

  d_Dst[y * imageW + x] = sum;  
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
  *m_Input,
  *h_Buffer,
  *h_OutputCPU, 
  *m_OutputGPU,
  *d_Buffer;
  dim3 block_dimensions;
  dim3 grid_dimensions; 
  int imageW;
  int imageH;
  int shared_memory_size_horizontal;
  int shared_memory_size_vertical;
  double accuracy = ACCURACY;
  unsigned int i;
  clock_t h_start, h_end, d_start, d_end;
  double h_time_used, d_memcpy_time_used, d_computation_time_used;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);
  //filter_radius = 16;

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  //imageW = 16384;
  imageH = imageW;


  //Unified Host - Device Memory (On-Demand memcpy())
  cudaMallocManaged(&m_Input, imageW * imageH * sizeof(double));
  cudaMallocManaged(&m_OutputGPU, imageW * imageH * sizeof(double));
  
  // Allocate memory on host
  h_Filter    = (double *) malloc(FILTER_LENGTH * sizeof(double));
  h_Buffer    = (double *) malloc(imageW * imageH * sizeof(double));
  h_OutputCPU = (double *) malloc(imageW * imageH * sizeof(double));


  // Check malloc() for host
  if(!h_Filter || !m_Input || !h_Buffer || !h_OutputCPU || !m_OutputGPU)
    return -1;

  // Seed for Random Numbers
  srand(200);

  for (i = 0; i < FILTER_LENGTH; i++) {
    h_Filter[i] = (double)(rand() % 16);
  }

  for (i = 0; i < imageW * imageH; i++) {
    m_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
  }

#ifdef CPU

  // Start CPU computation
  printf("\033[0;31mCPU computation...\033[0m\n");

  //Start calculate CPU time
  h_start = clock();
 
  convolutionRowCPU(h_Buffer, m_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
 
  //End calculate CPU time
  h_end = clock();
  h_time_used = ((double) (h_end - h_start)) / CLOCKS_PER_SEC;

#endif

  // Allocate memory on device
  cudaMalloc((void **) &d_Buffer, imageW * imageH * sizeof(double));

  //Check malloc() for device
  if(!d_Buffer)
    return -1;

 
  //Initialize Grid
  grid_init(&grid_dimensions, &block_dimensions, imageW, imageH);
  printf("Grid(%d, %d), Block(%d, %d)\n", grid_dimensions.x, grid_dimensions.y, block_dimensions.x, block_dimensions.y);
  
  // Start GPU computation
  printf("\033[0;32mGPU computation...\033[0m\n");
  
  //Start calculate GPU memory copy time
  d_start = clock();

  // Copying filter from host to constant memory
  cudaMemcpyToSymbol(c_Filter, h_Filter, FILTER_LENGTH * sizeof(double)); //Copy Filter to Constant Memory
  
  d_end = clock();
  d_memcpy_time_used = ((double) (d_end - d_start)) / CLOCKS_PER_SEC;


  d_start = clock();
  
  //Shared Memory Size
  shared_memory_size_horizontal = block_dimensions.y * (block_dimensions.x + 2 * filter_radius);
  shared_memory_size_vertical = (block_dimensions.y + 2 * filter_radius) * block_dimensions.x;

  //Kernel call
  convolutionRowGPU<<<grid_dimensions, block_dimensions, shared_memory_size_horizontal * sizeof(double)>>>(d_Buffer, m_Input, imageW, imageH, filter_radius); //First Kernel must finish copying data
  convolutionColumnGPU<<<grid_dimensions, block_dimensions, shared_memory_size_vertical * sizeof(double)>>>(m_OutputGPU, d_Buffer, imageW, imageH, filter_radius); //After copying is finished launch
  cudaDeviceSynchronize();
  cudaCheckError();

  //End calculate GPU time
  d_end = clock();
  d_computation_time_used = ((double) (d_end - d_start)) / CLOCKS_PER_SEC;

  
                                                                              

  
#ifdef CPU

  // Compare CPU and GPU results
  for(int i = 0; i < imageH * imageW; i++){
    if(ABS(h_OutputCPU[i] - m_OutputGPU[i]) > accuracy)
      accuracy = ABS(h_OutputCPU[i] - m_OutputGPU[i]);
  }



  printf("Comparing CPU and GPU with Accuracy %lf\n", accuracy);
  printf("\033[0;31mCPU Computation = %lf\033[0m\n", h_time_used);

#endif

  
  printf("\033[0;32mGPU Time = Memory Copy(%lf) + Computation(%lf) = %lf\033[0m\n", d_memcpy_time_used, d_computation_time_used, d_memcpy_time_used + d_computation_time_used);
  
  
  // Free all the allocated host memory ja
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Filter);

  // Free all the allocated device memory
  cudaFree(d_Buffer);

  // Free all the allocated unified memory 
  cudaFree(m_OutputGPU);
  cudaFree(m_Input);


  // Do a device reset just in case...
  cudaDeviceReset();
  
  return 0;
}
