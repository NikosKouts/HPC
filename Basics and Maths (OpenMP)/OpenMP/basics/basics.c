#include <stdio.h>
#include <omp.h>
#include <time.h>




int main(int argc, char const *argv[]){
  int processors, maximum_threads, total_threads, thread_id;
  clock_t start, end;
  double cpu_time_used;

  // Start of execution
  start = clock();

  // Get how many processors the computer running the program has
  processors = omp_get_num_procs();
  printf("Total Logical Processors (Hyperthreading Enabled): %d\n", processors);

  // Get maximum number of computer threads
  maximum_threads = omp_get_max_threads();
  printf("Maximum Number of Threads: %d\n", maximum_threads);

  // Get number of threads participating in the execution
  total_threads = omp_get_num_threads();
  printf("Total Threads Participating: %d\n", total_threads);

  
  #pragma omp parallel 
  {
    // Get thread id
    thread_id = omp_get_thread_num();
    
    // Master thread
    if(!thread_id)
      printf("Hello. I am the master thread.\n");
    // Slave threads
    else
      printf("Hi there! I am thread %d\n", thread_id);
  }


  // End of execution
  end = clock();

  // Calculate time used for the whole execution
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  fprintf(stderr, "Time of Execution: %lf\n", cpu_time_used);


  return 0;
}
