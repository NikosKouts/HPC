#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define N 99999999 // 10^8-1


int main(int argc, char *argv[]){
  int size, rank;
  double dx = (double) 1 / N;
  double *xi_array;
  double *private_xi_array;
  double y = 0;
  double pi;
  double current_time, comm_time, start, max_time, comp_time;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(!rank){
    xi_array = (double *) malloc((N + 1) * sizeof(double));
    if(!xi_array){
      exit(EXIT_FAILURE);
    }

    // Calculate different xi values
    for(int i = 0; i < N + 1; i++){
      xi_array[i] = i * dx;
    }    
  }
  
  // Thread private array to store xi variables
  private_xi_array = (double *) malloc(((N + 1) / size) * sizeof(double));
  if(!private_xi_array){
    exit(EXIT_FAILURE);
  }

  start = MPI_Wtime();

  // Scatter sub-arrays to all nodes
  MPI_Scatter(xi_array, (N + 1) / size, MPI_DOUBLE, private_xi_array, (N + 1) / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  current_time = MPI_Wtime() - start;
  MPI_Reduce(&current_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  comm_time = max_time;
  MPI_Barrier;
  
  start = MPI_Wtime();
  for(int i = 0; i < (N + 1) / size; i++){
    if(rank * ((N + 1) / size) + i == 0 || (rank * ((N + 1) / size) + i) == N){
      y += (4 / (1 + private_xi_array[i] * private_xi_array[i]));
    }
    else if((rank * ((N + 1) / size) + i) % 2 != 0){
      y += 4 * (4 / (1 + private_xi_array[i] * private_xi_array[i]));
    }
    else {
      y += 2 * (4 / (1 + private_xi_array[i] * private_xi_array[i]));
    }
  }
  y = y * (dx / 3);

  current_time = MPI_Wtime() - start;
  printf("id %d, curr time %f\n", rank, current_time);
  MPI_Reduce(&current_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  comp_time = max_time;
  start = MPI_Wtime();

  // Reduction of partial sums to master thread
  MPI_Reduce(&y, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  current_time = MPI_Wtime() - start;
  MPI_Reduce(&current_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  comm_time += max_time;


  if(!rank){
    printf("Value of pi = %.16lf with N = %d\n", pi, N);
    printf("pi value is almost %.20lf, MFLOPS = %4.3lf\n", pi, (2*(N+1)/size) / comp_time / 1000000);
    fprintf(stderr, "Threads: %d, Computation Time: %lf, Communication Time: %lf\n", size, comp_time, comm_time);
  }



  // Finalize the MPI environment.
  MPI_Finalize();

  
  return 0;
}
