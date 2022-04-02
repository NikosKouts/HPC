#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#define N 24000
#define CHUNK_SIZE 1000

int main(int argc, char const *argv[]){
  int processors, i, thread_id;
  double *results;
  double x, y, pi;
  clock_t start, end;
  double cpu_time_used;
  double FLOPS;

  // Get total number of logical processors  
  processors = omp_get_num_procs();
  
  /*
    Results array is used to accumulate the result of each thread in a separate cell.
    This is important in case the N is larger than the available processors and a 
    parallel for should be used. Each processors does multiple calculations in this
    case instead of one. Syncronization with atomic operations or locks is avoided since
    each thread has each own memory cell to work with.
  */
  results = (double *) calloc(processors, sizeof(double));
  if(!results)
    exit(EXIT_FAILURE);

  // Start of computation
  start = clock();

  #pragma omp parallel for private(i, x, y, thread_id) schedule(dynamic, CHUNK_SIZE)
  for(i = 0; i < N + 1; i++){
    // Get current thread to write on the correct position in the results array
    thread_id = omp_get_thread_num();


    // Computation
    x = (double) i / N;
    y = 4 / (1 + (x * x));

    // Edges
    if(i == 0 || i == N){
      results[thread_id] += y;
    }
    // Odd 
    else if(i % 2 != 0){
      results[thread_id] += 4 * y;
    }
    // Even
    else { 
      results[thread_id] += 2 * y;
    }
  }

  //Accumulate result of each thread
  for(i = 0; i < processors; i++){
    pi += results[i];
  }
  pi /= (3 * N);

  // End of execution
  end = clock();

  // Calculate time used for the whole execution
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  FLOPS = 8 * N /  cpu_time_used + processors + 1;

  printf("pi is approximately %lf , computations time = %lf, number of threads = %d, FLOPS = %lf, chunk = %d, scheduling = dynamic\n", pi, cpu_time_used, omp_get_num_procs(), FLOPS, CHUNK_SIZE);
  fprintf(stderr, "%d %lf\n", omp_get_max_threads(), cpu_time_used);

  return 0;
}
