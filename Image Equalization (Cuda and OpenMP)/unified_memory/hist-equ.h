#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#define CPU
#define TOTAL_TILES(IMG_SIZE) (IMG_SIZE / (8192 * 1024) + (IMG_SIZE % (8192 * 1024) != 0))
#define STREAMS 16
#define HISTOGRAM_SHARED_MEMORY

#define cudaCheckError() {                                                                        \
  cudaError_t error=cudaGetLastError();                                                           \
  if(error!=cudaSuccess) {                                                                        \
    fprintf(stderr, "ERROR IN CUDA %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));   \
    cudaDeviceReset();                                                                            \
    exit(EXIT_FAILURE);                                                                           \
  }                                                                                               \
}


typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    



PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

void gpu_equalization(int * hist_out, unsigned char *img_in, unsigned char * img_out, int img_size, int nbr_bin);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);


#endif
