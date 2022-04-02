#include "header.h"
#define DIMENSIONS 100
     
     
    
int main(int argc, char const *argv[]){
  double **array;
  clock_t start, end;
  double cpu_time_used;
  double norm = 0;
  int n, k;

  //Default
  if(argc != 3){
    n = 100;
    k = 2;
  }
  else {
    n = atoi(argv[1]);
    k = atoi(argv[2]);
  }

  //Create array
  array = array_constructor(n, NUMBER);
  
  //Calculating First Norm
  start = clock();
  norm = first_norm(array, n, k);
  if(norm < 0)
    return 0;
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("First Norm Time: %lf\n", cpu_time_used);
  fprintf(stderr, "First Norm: %lf\n", norm);


  //Calculating Infinite Norm
  start = clock(); 
  norm = infinite_norm(array, n, k);
  if(norm < 0)
    return 0;
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Infinite Norm Time: %lf\n", cpu_time_used);

  fprintf(stderr, "Infinite Norm: %lf\n", norm);
  return 0;
}
