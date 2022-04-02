#include <stdio.h>
#include <mpi.h>
#include <string.h>

#define SIZE 256

int main(int argc, char** argv) {
  int size, rank;
  MPI_Status status;
  char message[SIZE] = "Hello. I am the master node.";
  char buffer[SIZE];
  char response[SIZE];
  int receivers = 0;
  double start, end;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Master thread
  if(!rank){
    printf("Number of tasks participating in the execution: %d\n", size);
    start = MPI_Wtime();
    printf("%s\n", message);

    // Send message to slaves
    for(int i = 1; i < size; i++){
      MPI_Send(message, strlen(message) + 1, MPI_BYTE, i, 0, MPI_COMM_WORLD);
    }
    
    // Receive confirmation from slaves
    for(int i = 1; i < size; i++){
      MPI_Recv(response, SIZE, MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    }

    printf("%d nodes replied back to master node.\n", size - 1);

    end = MPI_Wtime();
    printf("Total execution time: %lf seconds\n", end - start);
    fprintf(stderr, "For %d Processes the Execution Time is: %lf\n", size, end - start);
  }
  // Other threads
  else {
    MPI_Recv(buffer, SIZE, MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    
    // Create and print response
    sprintf(response, "Hello. This is node %d.", rank);
    printf("%s\n", response);
    
    // Send response to master thread
    MPI_Send(response, strlen(response) + 1, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
  }

  // Finalize the MPI environment.
  MPI_Finalize();
}