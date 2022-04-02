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
// #define CPU

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

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, int i, int j, int image_dimensions, int tile_width, int filterR){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;
  int d;


  // Padded Shared Memory
  extern __shared__ double s_src[];

  // Left Padding
  if(tx < filterR){
    s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx] = d_Src[(y + filterR) * tile_width + x];
  }

  //Right Padding
  if(tx + filterR > BLOCK_DIMENSION - 1){
    s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx + 2 * filterR] = d_Src[(y + filterR) * tile_width + x + 2 * filterR];
  }


  //Actual Value
  s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + tx + filterR] = d_Src[(y + filterR) * tile_width + x + filterR];

  __syncthreads();


  //Actual Convolution
  for (k = -filterR; k <= filterR; k++) {
    d = tx + k;

    sum += s_src[ty * (BLOCK_DIMENSION + 2 * filterR) + d + filterR] * c_Filter[filterR - k];  
  }
  

  d_Dst[(y + i) * image_dimensions + x + j] = sum;
}


__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, int i, int j, int image_dimensions, int tile_width, int filterR){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0;
  int k;
  int d;

  // Padded Shared Memory
  extern __shared__ double s_src[];
  
  
  if(ty < filterR){
    s_src[ty * BLOCK_DIMENSION + tx] = d_Src[y * tile_width + x + filterR];
  }


  if(ty + filterR > BLOCK_DIMENSION - 1){
    s_src[(ty + 2 * filterR) * BLOCK_DIMENSION + tx] = d_Src[(y + 2 * filterR) * tile_width + x + filterR];
  }
  
  s_src[(ty + filterR) * BLOCK_DIMENSION + tx] = d_Src[(y + filterR) * tile_width + x + filterR];

  __syncthreads();


  for (k = -filterR; k <= filterR; k++) {
    d = ty + k;
  
    sum += s_src[(d + filterR) * BLOCK_DIMENSION + tx] * c_Filter[filterR - k];
  }

  d_Dst[(y + i) * image_dimensions + x + j] = sum;  
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



// Function that performs a tiled row convolution and returns a new image that is located on device for further processing
double *row_tiling_convolution(double *h_input, int image_dimensions, dim3 block_dimensions, dim3 grid_dimensions, int tile_count){
  double **h_tiles, **d_tiles;
  int tile_data_width, tile_data_height;
  int tile_width, tile_height;
  int shared_memory_size_horizontal;
  double *m_buffer;  
  int tile = 0;
  cudaStream_t *streams;
  cudaError_t *results;

  int total_tiles = tile_count * tile_count;

  // Set tile's Dimensions
  tile_data_width = tile_data_height = image_dimensions / tile_count;
  tile_width = tile_height = tile_data_width + 2 * filter_radius;

  // Buffer memory on device
  cudaMallocManaged((void **) &m_buffer, image_dimensions * image_dimensions * sizeof(double));
  if(!m_buffer)
    exit(EXIT_FAILURE);


  //Array of tiles to support async copies
  h_tiles = (double **) malloc(total_tiles * sizeof(double *));
  if(!h_tiles)
    exit(EXIT_FAILURE);
  
  // Allocate memory for each tile
  for(int i = 0; i < total_tiles; i++){
    h_tiles[i] = (double *) calloc(tile_height * tile_width, sizeof(double));
    if(!h_tiles[i])
      exit(EXIT_FAILURE);
  }

  // Create Streams
  streams = (cudaStream_t *) malloc(total_tiles * sizeof(cudaStream_t));
  if(!streams)
    exit(EXIT_FAILURE);

  //Create Results Array
  results = (cudaError_t *) malloc(total_tiles * sizeof(cudaError_t));
  if(!results)
    exit(EXIT_FAILURE);
  
  // Initialize Streams and Results
  for(int i = 0; i < total_tiles; i++){
    results[i] = cudaStreamCreate(streams + i);
  }


  // Host array of pointers to device arrays
  d_tiles = (double **) malloc(total_tiles * sizeof(double *));
  if(!d_tiles)
    exit(EXIT_FAILURE);
  
  // Allocate memory for each device tile
  for(int i = 0; i < total_tiles; i++){
    cudaMalloc((void **) &d_tiles[i], tile_height * tile_width * sizeof(double));
    if(!d_tiles[i])
      exit(EXIT_FAILURE);
  }


  //Shared memory size for row convolution
  shared_memory_size_horizontal = block_dimensions.y * (block_dimensions.x + 2 * filter_radius);

  //Handle tiling, memory copies, and allocating memory on streams 
  for(int i = 0; i < image_dimensions; i += tile_data_height){
    for(int j = 0; j < image_dimensions; j += tile_data_width){
      
      // Steps inside block
      for(int ii = 0; ii < tile_data_height; ii++){
        for(int jj = 0; jj < tile_data_width; jj++){

          // Set Top and Bottom Respectively
          if(i != 0 && ii < filter_radius){
            h_tiles[tile][ii  * tile_width + jj + filter_radius] = h_input[(i + ii - filter_radius) * image_dimensions + (j + jj)];
          }
          else if(i < image_dimensions - tile_data_height && ii >= tile_data_height - filter_radius){
            h_tiles[tile][(ii + 2 * filter_radius)  * tile_width + jj + filter_radius] = h_input[(i + ii + filter_radius) * image_dimensions + (j + jj)];
          }
          

          // Set Left and Right Respectively
          if(j != 0 && jj < filter_radius){
            h_tiles[tile][(ii + filter_radius) * tile_width + jj] = h_input[(i + ii) * image_dimensions + (j + jj - filter_radius)];
          }
          else if(j < image_dimensions - tile_data_width && jj >= tile_data_width - filter_radius){
            h_tiles[tile][(ii + filter_radius) * tile_width + jj + 2 * filter_radius] = h_input[(i + ii) * tile_data_width + (j + jj + filter_radius)];;
          }

          // Set Current Postition
          h_tiles[tile][(ii + filter_radius) * tile_width + (jj + filter_radius)] = h_input[(i + ii) * image_dimensions + (j + jj)]; 
        }
      }
      

      // Copy memory to device
      cudaMemcpyAsync(d_tiles[tile], h_tiles[tile], tile_height * tile_width * sizeof(double), cudaMemcpyHostToDevice, streams[tile]);


      // printf("\n----------------- [ Tile %d %d ] -----------------\n", i, j);
      // for(int row = 0; row < tile_height; row++){
      //   for(int col = 0; col < tile_width; col++){
      //     printf("%lf ", h_tiles[tile][row * tile_width + col]);
      //   }
      //   printf("\n");
      // }
      tile++;  
    }
  }


  // Convolution on seperate streams
  tile = 0;
  for(int i = 0; i < image_dimensions; i += tile_data_height){
    for(int j = 0; j < image_dimensions; j += tile_data_width){
       // Start GPU convolution 
      convolutionRowGPU<<<grid_dimensions, block_dimensions, shared_memory_size_horizontal * sizeof(double), streams[tile]>>>(m_buffer, d_tiles[tile], i, j, image_dimensions, tile_width, filter_radius);
      cudaCheckError();

      tile++;
    }
  }
  
  cudaDeviceSynchronize();

  //Destroy Streams
  for(int i = 0; i < total_tiles; i++){
    results[i] = cudaStreamDestroy(streams[i]);
  }

  return m_buffer;
}



// Function that performs a tiled row convolution and returns a new image that is located on device for further processing
double *column_tiling_convolution(double *h_input, int image_dimensions, dim3 block_dimensions, dim3 grid_dimensions, int tile_count){
  double **h_tiles, **d_tiles;
  int tile_data_width, tile_data_height;
  int tile_width, tile_height;
  int shared_memory_size_vertical;
  double *m_output_GPU;
  cudaStream_t *streams;
  cudaError_t *results;
  int tile = 0;

  int total_tiles = tile_count * tile_count;

  // Set tile's Dimensions
  tile_data_width = tile_data_height = image_dimensions / tile_count;
  tile_width = tile_height = tile_data_width + 2 * filter_radius;

  //Unified Host - Device Memory (On-Demand memcpy())
  cudaMallocManaged(&m_output_GPU, image_dimensions * image_dimensions * sizeof(double));
  if(!m_output_GPU)
    exit(EXIT_FAILURE);


  //Array of tiles to support async copies
  h_tiles = (double **) malloc(total_tiles * sizeof(double *));
  if(!h_tiles)
    exit(EXIT_FAILURE);
  
  // Allocate memory for each tile
  for(int i = 0; i < total_tiles; i++){
    h_tiles[i] = (double *) calloc(tile_height * tile_width, sizeof(double));
    if(!h_tiles[i])
      exit(EXIT_FAILURE);
  }

  // Create Streams
  streams = (cudaStream_t *) malloc(total_tiles * sizeof(cudaStream_t));
  if(!streams)
    exit(EXIT_FAILURE);

  //Create Results Array
  results = (cudaError_t *) malloc(total_tiles * sizeof(cudaError_t));
  if(!results)
    exit(EXIT_FAILURE);
  
  // Initialize Streams and Results
  for(int i = 0; i < total_tiles; i++){
    results[i] = cudaStreamCreate(streams + i);
  }


  // Host array of pointers to device arrays
  d_tiles = (double **) malloc(total_tiles * sizeof(double *));
  if(!d_tiles)
    exit(EXIT_FAILURE);
  
  // Allocate memory for each device tile
  for(int i = 0; i < total_tiles; i++){
    cudaMalloc((void **) &d_tiles[i], tile_height * tile_width * sizeof(double));
    if(!d_tiles[i])
      exit(EXIT_FAILURE);
  }


  // Shared memory size for column convolution
  shared_memory_size_vertical = (block_dimensions.y + 2 * filter_radius) * block_dimensions.x;


  for(int i = 0; i < image_dimensions; i += tile_data_height){
    for(int j = 0; j < image_dimensions; j += tile_data_width){
      
      // Steps inside block
      for(int ii = 0; ii < tile_data_height; ii++){
        for(int jj = 0; jj < tile_data_width; jj++){

          // Set Top and Bottom Respectively
          if(i != 0 && ii < filter_radius){
            h_tiles[tile][ii  * tile_width + jj + filter_radius] = h_input[(i + ii - filter_radius) * image_dimensions + (j + jj)];
          }
          else if(i < image_dimensions - tile_data_height && ii >= tile_data_height - filter_radius){
            h_tiles[tile][(ii + 2 * filter_radius)  * tile_width + jj + filter_radius] = h_input[(i + ii + filter_radius) * image_dimensions + (j + jj)];
          }
          

          // Set Left and Right Respectively
          if(j != 0 && jj < filter_radius){
            h_tiles[tile][(ii + filter_radius) * tile_width + jj] = h_input[(i + ii) * image_dimensions + (j + jj - filter_radius)];
          }
          else if(j < image_dimensions - tile_data_width && jj >= tile_data_width - filter_radius){
            h_tiles[tile][(ii + filter_radius) * tile_width + jj + 2 * filter_radius] = h_input[(i + ii) * tile_data_width + (j + jj + filter_radius)];;
          }

          // Set Current Postition
          h_tiles[tile][(ii + filter_radius) * tile_width + (jj + filter_radius)] = h_input[(i + ii) * image_dimensions + (j + jj)]; 
        }
      }
      
       // Copy memory to device
      cudaMemcpyAsync(d_tiles[tile], h_tiles[tile], tile_height * tile_width * sizeof(double), cudaMemcpyHostToDevice, streams[tile]);

      cudaCheckError();     
      
      tile++; 
    }
  }

  // Convolution on seperate streams
  tile = 0;
  for(int i = 0; i < image_dimensions; i += tile_data_height){
    for(int j = 0; j < image_dimensions; j += tile_data_width){
      // Start GPU convolution 
      convolutionColumnGPU<<<grid_dimensions, block_dimensions, shared_memory_size_vertical * sizeof(double), streams[tile]>>>(m_output_GPU, d_tiles[tile], i, j, image_dimensions, tile_width, filter_radius);

      cudaCheckError();

      tile++;
    }
  }

  cudaDeviceSynchronize();

  //Destroy Streams
  for(int i = 0; i < total_tiles; i++){
    results[i] = cudaStreamDestroy(streams[i]);
  }


  return m_output_GPU;
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
    *m_OutputGPU,
    *m_buffer,
    accuracy = ACCURACY,
    h_time_used,
    d_memcpy_time_used,
    d_computation_time_used;
  dim3 
    block_dimensions,
    grid_dimensions; 
  int 
    imageW,
    imageH,
    tile_count,
    tile_data_width,
    tile_data_height;
  unsigned int i;
  clock_t h_start, h_end, d_start, d_end;


	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  imageH = imageW;

  printf("Enter the amount of tiles to split image: ");
  scanf("%d", &tile_count);

  
  // Allocate memory on host
  h_Filter    = (double *) malloc(FILTER_LENGTH * sizeof(double));
  h_Input     = (double *) malloc(imageW * imageH * sizeof(double));
  h_Buffer    = (double *) malloc(imageW * imageH * sizeof(double));
  h_OutputCPU = (double *) malloc(imageW * imageH * sizeof(double));


  // Check malloc() for host
  if(!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU)
    return -1;

  // Seed for Random Numbers
  srand(200);

  for (i = 0; i < FILTER_LENGTH; i++) {
    //h_Filter[i] = (double)(rand() % 16);
    h_Filter[i] = 1;
  }

  for (i = 0; i < imageW * imageH; i++) {
    //h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    h_Input[i] = 1;
  }

#ifdef CPU

  // Start CPU computation
  printf("\033[0;31mCPU computation...\033[0m\n");

  //Start calculate CPU time
  h_start = clock();
 
  convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
 
  //End calculate CPU time
  h_end = clock();
  h_time_used = ((double) (h_end - h_start)) / CLOCKS_PER_SEC;

#endif

  //********************************************** GPU **********************************************// 
  
  // Set tile's Dimensions
  tile_data_width = tile_data_height = imageW / tile_count;
  

  //Initialize Grid
  grid_init(&grid_dimensions, &block_dimensions, tile_data_width, tile_data_height);
  printf("Grid(%d, %d), Block(%d, %d)\n", grid_dimensions.x, grid_dimensions.y, block_dimensions.x, block_dimensions.y);
  
  // Start GPU computation
  printf("\033[0;32mGPU computation...\033[0m\n");
  
  //Start calculate GPU memory copy time
  d_start = clock();

  // Copying filter from host to constant memory
  cudaMemcpyToSymbol(c_Filter, h_Filter, FILTER_LENGTH * sizeof(double), 0, cudaMemcpyHostToDevice);
  
  d_end = clock();
  d_memcpy_time_used = ((double) (d_end - d_start)) / CLOCKS_PER_SEC;


  d_start = clock();

  

  //Convolution Calculation
  m_buffer = row_tiling_convolution(h_Input, imageW, block_dimensions, grid_dimensions, tile_count);
  m_OutputGPU = column_tiling_convolution(m_buffer, imageW,  block_dimensions, grid_dimensions, tile_count);
  
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
  free(h_Input);
  
  // Free all the allocated device memory
  cudaFree(m_buffer);

  // Free all the allocated unified memory 
  cudaFree(m_OutputGPU);
  

  // Do a device reset just in case...
  cudaDeviceReset();
  
  return 0;
}
