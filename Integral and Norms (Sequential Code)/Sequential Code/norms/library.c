#include "header.h"


//Initialize Array
void array_init(double **array, int n, double number){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++)
      array[i][j] = number;
  }
}

//Construct a 2D array with dimensions n and initialize it with number
double **array_constructor(int n, double number){
  double **array;

  array = (double **) malloc(n * sizeof(double *));
  if(!array)
    return NULL;
  
  for(int i = 0; i < n; i++){
    array[i] = (double *) malloc(n * sizeof(double));
    if(!array[i])
      return NULL;
  }

  // Initialize array with number
  array_init(array, n, number);


#ifdef DEBUG
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      printf("%lf ", array[i][j]);
    }
    printf("\n");
  }
#endif


  return array;
}


double first_norm(double **array, int n, int k){
  double sum = 0;
  double max = -1;

  if(k > n)
    return -1;

  for(int j = 0; j < n; j++){ 
    for(int i = 0; i < n; i += k){
     /* for(int step = 0; (step < k) && (i + step < n); step++)
        sum += array[i + step][j];*/

      sum += array[i][j];
        
      if(k==2)
        sum += array[i+1][j];
      else if(k==4)
        sum += array[i+1][j] + array[i+2][j] + array[i+3][j];
      else if(k==8)
        sum += array[i+1][j] + array[i+2][j] + array[i+3][j] + array[i+4][j] + array[i+5][j] + array[i+6][j] + array[i+7][j];
      else if(k==16){
        sum += array[i+1][j] + array[i+2][j] + array[i+3][j] + array[i+4][j] + array[i+5][j] + array[i+6][j] + array[i+7][j]
            + array[i+8][j] + array[i+9][j] + array[i+10][j] + array[i+11][j] + array[i+12][j] + array[i+13][j] + array[i+14][j] + array[i+15][j];
      }
    }

    if(sum > max)
      max = sum;
    sum = 0;
  }
  
  return max;
}

double infinite_norm(double **array, int n, int k){
  double sum = 0;
  double max = -1;

  if(k > n)
    return -1;

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j += k){ 
    
     /* for(int step = 0; (step < k) && (i + step < n); step++)
        sum += array[i + step][j];*/

      sum += array[i][j];
        
      if(k==2)
        sum += array[i][j+1];
      else if(k==4)
        sum += array[i][j+1] + array[i][j+2] + array[i][j+3];
      else if(k==8)
        sum += array[i][j+1] + array[i][j+2] + array[i][j+3] + array[i][j+4] + array[i][j+5] + array[i][j+6] + array[i][j+7];
      else if(k==16){
        sum += array[i][j+1] + array[i][j+2] + array[i][j+3] + array[i][j+4] + array[i][j+5] + array[i][j+6] + array[i][j+7]
            + array[i][j+8] + array[i][j+9] + array[i][j+10] + array[i][j+11] + array[i][j+12] + array[i][j+13] + array[i][j+14] + array[i][j+15];
      }
    }

    if(sum > max)
      max = sum;
    sum = 0;
  }
  
  return max;
}