#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define N 100000000 // Array Size
#define M 40       // Jacobi iterations

int main(int argc, char *argv[])
{

  int size;
  int rank;
  float *x_old, *x_new, *b;
  double residual, difference;
  float *private_x_new, *private_b, *private_x_old;
  int *sendcounts, *displs;
  double private_residual, private_difference;
  int i, j, l;
  double start_time, max_time, current_time;
  double comp_time, comm_time;
  int ierr;
  int thread_workload;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  thread_workload = N / size;

  // allocate memory for subvectors
  private_x_new = (float *) malloc((thread_workload + 2) * sizeof(float));
  private_x_old = (float *) malloc((thread_workload + 2) * sizeof(float));
  private_b = (float *) malloc(thread_workload * sizeof(float));
  x_old = (float *) malloc(N * sizeof(float));
  
  // Master thread
  if (rank == 0){
   
    x_new = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));

    
    for (i = 0; i < N; i++){
      x_new[i] = 0.0;
      x_old[i] = (float) 1;
      b[i] = 0.0;
    }
    b[N - 1] = (double)N + 1;

    // Parameters for scattering data
    sendcounts = (int *) malloc(size * sizeof(int));
    displs = (int *) malloc(size * sizeof(int));
    if (size == 1){
      sendcounts[0] = N;
      displs[0] = 0;
    }
    else if (size == 2){
      sendcounts[0] = N / 2 + 1;
      sendcounts[1] = N / 2 + 1;
      displs[0] = 0;
      displs[1] = N / 2 - 1;
    }
    else {
      displs[0] = 0;
      sendcounts[0] = thread_workload + 1;
      for (i = 1; i < size - 1; i++){
        sendcounts[i] = thread_workload + 2;
        displs[i] = i * thread_workload - 1;
      }
      sendcounts[size - 1] = thread_workload + 1;
      displs[size - 1] = N - thread_workload - 1;
    }
  }

  // Communication timer start
  if (rank == 0)
    start_time = MPI_Wtime();

  // Scatter b to private_b
  MPI_Scatter(b, thread_workload, MPI_FLOAT, private_b, thread_workload, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Communication timer stop
  if(!rank){
    comm_time = MPI_Wtime() - start_time;
  }

  // Jacobi 
  for (i = 0; i < M; i++){
    if (!rank)
      start_time = MPI_Wtime();
    
    MPI_Scatterv(x_old, sendcounts, displs, MPI_FLOAT, private_x_old, thread_workload + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(!rank)
      comm_time += MPI_Wtime() - start_time;
    
    start_time = MPI_Wtime();

    // Compute x_new
    if(!rank){
      private_x_new[0] = (0.5) * (private_x_old[1] + private_b[0]);
      for (j = 1; j < thread_workload; j++)
        private_x_new[j] = (0.5) * (private_x_old[j - 1] + private_x_old[j + 1] + private_b[j]);
    }
    else if(rank == size - 1){
      private_x_new[thread_workload - 1] = (0.5) * (private_x_old[thread_workload - 2] + private_b[thread_workload - 1]);
      for (j = 0; j < thread_workload - 1; j++)
        private_x_new[j] = (0.5) * (private_x_old[j] + private_x_old[j + 2] + private_b[j]);
    }
    else {
      for (j = 0; j < thread_workload; j++)
        private_x_new[j] = (0.5) * (private_x_old[j] + private_x_old[j + 2] + private_b[j]);
    }
    current_time = MPI_Wtime() - start_time;
    MPI_Reduce(&current_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!rank){
      comp_time = max_time;
      start_time = MPI_Wtime();
    }

    MPI_Gather(private_x_new, thread_workload, MPI_FLOAT, x_new, thread_workload, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(x_new, sendcounts, displs, MPI_FLOAT, private_x_new, thread_workload + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (!rank){
      comm_time += MPI_Wtime() - start_time;
      start_time = MPI_Wtime();
    }

    // Compute residual
    private_residual = 0;
    if(!rank){
      private_residual = fabs(2.0 * private_x_new[0] - private_x_new[1] - private_b[0]);
      for (j = 1; j < thread_workload; j++)
        private_residual += fabs(2.0 * private_x_new[j] - private_x_new[j - 1] - private_x_new[j + 1] - private_b[0]);
    }
    else if(rank == size - 1){
      private_residual = fabs(-private_x_new[thread_workload - 1] + 2.0 * private_x_new[thread_workload] - private_b[thread_workload - 1]);
      for (j = 0; j < thread_workload - 1; j++)
        private_residual += fabs(-private_x_new[j] + 2.0 * private_x_new[j + 1] - private_x_new[j + 2] - private_b[j]);
    }
    else {
      for (j = 0; j < thread_workload; j++)
        private_residual += fabs(-private_x_new[j] + 2.0 * private_x_new[j + 1] - private_x_new[j + 2] - private_b[j]);
    }

    current_time = MPI_Wtime() - start_time;
    MPI_Reduce(&current_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!rank){
      comp_time += max_time;
      start_time = MPI_Wtime();
    }

    MPI_Scatter(x_old, thread_workload, MPI_FLOAT, private_x_old, thread_workload, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(x_new, thread_workload, MPI_FLOAT, private_x_new, thread_workload, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (!rank){
      comm_time += MPI_Wtime() - start_time;
      start_time = MPI_Wtime();
    }
    // Difference 
    private_difference = 0;
    for (j = 0; j < thread_workload; j++){
      private_difference += pow(private_x_old[j] - private_x_new[j], 2);
    }

    current_time = MPI_Wtime() - start_time;
    MPI_Reduce(&current_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   
    if (!rank){
      comp_time += max_time;
      start_time = MPI_Wtime();
    }

    // send x_old, residual, back to root process
    MPI_Gather(private_x_new, thread_workload, MPI_FLOAT, x_old, thread_workload, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Reduce(&private_residual, &residual, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&private_difference, &difference, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!rank){
      comm_time += MPI_Wtime() - start_time;
      printf("iter = %d, residual = %.2f, difference = %.2f\n", i, residual, sqrt(difference));
    }
  }

  if (!rank){
    fprintf(stderr, "Threads: %d, Computation Time = %lf seconds, Communication Time = %lf seconds\n", size, comp_time, comm_time);
    free(x_old);
    free(x_new);
    free(b);
  }

  MPI_Finalize();


  return 0;
}
