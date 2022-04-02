#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "hist-equ.h"
#include <omp.h>
#define MAX_BLOCK_DIM 1024
#define HISTOGRAM_SIZE 256


// Constant Memory
__constant__ int c_histogram[HISTOGRAM_SIZE]; //Maximum Constant Memory


void grid_init(dim3 *grid_dimensions, dim3 *block_dimensions, int size){
	// Threads in block
	block_dimensions->x = size > MAX_BLOCK_DIM ? MAX_BLOCK_DIM : size;
	
	// Default grid dimensions
	grid_dimensions->x = 1;
	
	// More blocks in grid are needed
	if(size > MAX_BLOCK_DIM){
		grid_dimensions->x = size / MAX_BLOCK_DIM;
		if(size % MAX_BLOCK_DIM){
			grid_dimensions->x++;
		}
	}
}

__global__ void calculate_histogram(unsigned char *tile, int *histogram, int size){
	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef HISTOGRAM_SHARED_MEMORY
    // Store histogram to accelerate atomi operations
    __shared__ int s_histogram[HISTOGRAM_SIZE];
#endif

    //__shared__ unsigned char s_histogram[HISTOGRAM_SIZE];
	extern __shared__ unsigned char s_tile[];
	
    // Initialize histogram with zeroes 
    if(x < 256){
        histogram[x] = 0;
    }

    // Stay inside of tile
    if(x >= size){
		return;
	}

#ifdef HISTOGRAM_SHARED_MEMORY
    // Initialize share
    if(tx < 256){
        s_histogram[tx] = 0;
    }
#endif

    // Initialize tiles
	s_tile[tx] = tile[x];
	
    __syncthreads();

#ifdef HISTOGRAM_SHARED_MEMORY
    // Do calculations on shared memory to optimize for occupancy
	atomicAdd((unsigned int *) &s_histogram[(unsigned int) s_tile[tx]], 1);
    
    __syncthreads();
    

    // Bring back shared memory to global
    if(tx < 256){
       atomicAdd((unsigned int *) &histogram[tx], s_histogram[tx]);
    }
#else 
    // Do calculations on shared memory to optimize for occupancy
	atomicAdd((unsigned int *) &histogram[(unsigned int) s_tile[tx]], 1);
#endif
}

__global__ void equalize_image(unsigned char *tile, int size){
	int tx = threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned char s_tile[];
    if(x >= size){
        return;
	}

    s_tile[tx] = tile[x];
	__syncthreads();
    
    if(c_histogram[s_tile[tx]] > 255) {
        s_tile[tx] = 255;
    }
    else {
        s_tile[tx] = (unsigned char) c_histogram[s_tile[tx]];
    }

    tile[x] = s_tile[tx];
}

void gpu_equalization(int *hist_out, unsigned char *img_in, unsigned char * img_out, int img_size, int nbr_bin){
    int i, j, tile_index, stream_index, tile_size, size;
    unsigned char **d_tiles;
    cudaStream_t *streams;
	dim3 block_dimensions, grid_dimensions;
	size_t shared_memory_size;
	int **h_histograms, **d_histograms;
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int cdf = 0, min = 0, d;
    cudaEvent_t *start_events, *end_events; 
    int TILES = TOTAL_TILES(img_size);
    clock_t histogram_computation_start, histogram_computation_end;
    clock_t compute_new_histogram_start, compute_new_histogram_end;
    clock_t equalization_start, equalization_end;
    double total_histogram_computation;
    double total_compute_new_histogram;
    double total_equalization;

    // Set output histogram with zeroes
    memset(hist_out, 0, nbr_bin * sizeof(int));
    if(!hist_out){
        exit(EXIT_FAILURE);
    }
    
    // Calculate the size of each tile
    tile_size = img_size / TILES;

    // Check if tiles split the image perfectly
    if(img_size % TILES){
        //tile_size = (img_size / TILES) + 1;
        //printf("%d %d %lf\n",img_size, TILES, TOTAL_TILES(img_size));
        TILES++;
    }

    // Create an array of pointers to device's tiles
    d_tiles = (unsigned char **) malloc(TILES * sizeof(unsigned char *));
    if(!d_tiles){
        exit(EXIT_FAILURE);
    }

	// Histogram array of pointers to hold host's histograms for each tile
	h_histograms = (int **) malloc(TILES * sizeof(int *));
	if(!h_histograms){
		exit(EXIT_FAILURE);
	}	

    // Histogram array of pointers to hold device's histograms for each tile
    d_histograms = (int **) malloc(TILES * sizeof(int *));
	if(!d_histograms){
		exit(EXIT_FAILURE);
	}	

    // Create streams array
    streams = (cudaStream_t *) malloc(STREAMS * sizeof(cudaStream_t));
    if(!streams){
        exit(EXIT_FAILURE);
    }

    // Create start events array 
    start_events = (cudaEvent_t *) malloc(STREAMS * sizeof(cudaEvent_t));
    if(!start_events){
        exit(EXIT_FAILURE);
    }

    // Create end events array 
    end_events = (cudaEvent_t *) malloc(STREAMS * sizeof(cudaEvent_t));
    if(!end_events){
        exit(EXIT_FAILURE);
    }

   	// Initialize grid-block dimensions
   	grid_init(&grid_dimensions, &block_dimensions, tile_size);	


    // Allocate memory for chunks on device
    for(i = 0; i < TILES; i++){
        cudaMalloc((unsigned char **) &d_tiles[i], tile_size * sizeof(unsigned char));
        if(!d_tiles[i])
            exit(EXIT_FAILURE);
        
        // Error handling
        cudaCheckError();

        // Pin instances of histograms with DMA to RAM  
		cudaMallocHost((int **) &h_histograms[i], nbr_bin * sizeof(int));
		if(!h_histograms[i]){
			exit(EXIT_FAILURE);
		}

        // Error handling
        cudaCheckError();

        cudaMalloc((int **) &d_histograms[i], nbr_bin * sizeof(int));
		if(!d_histograms[i]){
			exit(EXIT_FAILURE);
		}

        // Error handling
        cudaCheckError();
    }

    for(i = 0; i < STREAMS; i++) {
        // Create streams
        cudaStreamCreate(&streams[i]);

        // Error handling
        cudaCheckError();

        // Create events
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);


        // Error handling
        cudaCheckError();
    }

	//Set shared memory size
	shared_memory_size = block_dimensions.x * sizeof(unsigned char);

    printf("Total Tiles %d with Tile Size: %d\n", TILES, tile_size);
    printf("Grid(%d), Block(%d)\n", grid_dimensions.x, block_dimensions.x);

    
    histogram_computation_start = clock();


#pragma omp parallel
{
    // Perform tiling to the image
    for(tile_index = 0; tile_index < TILES; tile_index += STREAMS){
        #pragma omp parallel for private(size)
        for(stream_index = 0; stream_index < STREAMS; stream_index++){
            if(tile_index + stream_index >= TILES){
                break;
            }

            // Calulate size of memory to copy to kernel
            size = tile_size;
            if(tile_index + stream_index == TILES - 1){
                size = img_size - (tile_index + stream_index) * tile_size;
            }

            // Start of kernel operations
            cudaEventRecord(start_events[stream_index], streams[stream_index]);
			cudaCheckError();

            // Copy tiles to device
            cudaMemcpyAsync(d_tiles[tile_index + stream_index], &img_in[(tile_index + stream_index) * tile_size], size, cudaMemcpyHostToDevice, streams[stream_index]);
			cudaCheckError();
            
            // Kernel Call
			calculate_histogram<<<grid_dimensions, block_dimensions, shared_memory_size, streams[stream_index]>>>(d_tiles[tile_index + stream_index], d_histograms[tile_index + stream_index], size);
			cudaCheckError();
            
            // Copy histograms back to host        
            cudaMemcpyAsync(h_histograms[tile_index + stream_index], d_histograms[tile_index + stream_index], nbr_bin * sizeof(int), cudaMemcpyDeviceToHost, streams[stream_index]);
			cudaCheckError();

            // End of kernel operations
            cudaEventRecord(end_events[stream_index], streams[stream_index]);
			cudaCheckError();    

            // Synchronize stream
            cudaEventSynchronize(end_events[stream_index]);
			cudaCheckError();        
        } 
    }

    // Merge histograms into one
    for(i = 0; i < TILES; i++){
        #pragma omp parallel for
        for(j = 0; j < 256; j++){ 
            hist_out[j] += h_histograms[i][j];
        }
    }
}

    histogram_computation_end = clock();
    total_histogram_computation = (double)(histogram_computation_end - histogram_computation_start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Total time to compute histogram %lf\n", total_histogram_computation);

    compute_new_histogram_start = clock();
    i = 0;
    while(min == 0){
        min = hist_out[i++];
    }

    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_out[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }   
    }
    
    compute_new_histogram_end = clock();
    total_compute_new_histogram = (double)(compute_new_histogram_end - compute_new_histogram_start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Total time to compute new histogram %lf\n", total_compute_new_histogram);


    equalization_start = clock();

    // Move new histogram to the constant memory
    cudaMemcpyToSymbol(c_histogram, lut, nbr_bin * sizeof(int), 0, cudaMemcpyHostToDevice);
    
    

#pragma omp parallel
{
    /* Get the result image */
    for(tile_index = 0; tile_index < TILES; tile_index += STREAMS){
        #pragma omp parallel for private(size)
		for(stream_index = 0; stream_index < STREAMS; stream_index++){
            if(tile_index + stream_index >= TILES){
                break;
            }

            // Start of kernel operations
            cudaEventRecord(start_events[stream_index], streams[stream_index]);
			cudaCheckError();

            //printf("%d %d\n", tile_index, stream_index);
			equalize_image<<<grid_dimensions, block_dimensions, shared_memory_size, streams[stream_index]>>>(d_tiles[tile_index + stream_index], tile_size);
			cudaCheckError();

            // Calculate size of memory to retrieve from kernel
            size = tile_size;
            if(tile_index + stream_index == TILES - 1){
                size = img_size - (tile_index + stream_index) * tile_size;
            }

            // Copy histograms back to host        
            cudaMemcpyAsync(&img_out[(tile_index + stream_index) * tile_size], d_tiles[tile_index + stream_index], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[stream_index]);
			cudaCheckError();

            // End of kernel operations
            cudaEventRecord(end_events[stream_index], streams[stream_index]);
			cudaCheckError();

            // Synchronize stream
            cudaEventSynchronize(end_events[stream_index]);
			cudaCheckError();   
		}		
    }    
}

    equalization_end = clock();
    total_equalization = (double)(equalization_end - equalization_start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Total time to equalize image %lf\n", total_equalization);


    for(i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(start_events[i]); 
    }

    for(i = 0; i < TILES; i++) {
        cudaFree(d_tiles[i]);
        cudaFree(d_histograms[i]);
        cudaFreeHost(h_histograms[i]);
    }

    free(d_histograms);
    free(h_histograms);
    free(lut);
    free(d_tiles);

    printf("GPU Finished\n");

}