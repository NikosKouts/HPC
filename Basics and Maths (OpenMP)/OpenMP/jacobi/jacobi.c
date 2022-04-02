/* Lantzos Stergios : 2789
 * Stoltidhs Alexandros: 2824
 * Koutsoukis Nikolaos: 2907
 */

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

#define M 20 //Jacobi iterations
#define N 10000//100000 array length


int main(int argc, char *argv[]) {
    
    //Variables
    float **A, **D_inverted, **L_plus_U;
	float *b, *x_old, *x_new, *x_temp;
    int i, j, k;
    float residual, difference;
    double start, end; //timers       
    
    /* STAGE: 1 */
    //Dynamic memory for our 2D - arrays
    A = (float**) malloc (N * sizeof(float*));
    if (A == NULL) {
        exit(-1);
    }
	D_inverted = (float**) malloc (N * sizeof(float*));
	if (D_inverted == NULL) {
		exit(-1);
	}
	L_plus_U = (float**) malloc (N * sizeof(float*));
	if (L_plus_U == NULL) {
		exit(-1);
	}
	
	//Dynamic memory for their sits
    for (i = 0; i < N; i++) {
        A[i] = (float*) malloc (N * sizeof(float));
		D_inverted[i] = (float*) malloc( N * sizeof(float));
		L_plus_U[i] = (float*) malloc (N * sizeof(float));
    }
    
    //Dynamic memory for our vectors
    b = (float *)malloc(N * sizeof(float));
    x_old = (float *) malloc(N * sizeof(float));
    x_new = (float *) malloc(N * sizeof(float));
    x_temp = (float *) malloc(N * sizeof(float));
    
    
    
    
    /* STAGE: 2 */
    
    //Set values for 2D arrays
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                A[i][j] = 2.0;
                D_inverted[i][j] = (float) 1/2;
            }
            else if (j == i+1) {
                A[i][j] = -1.0;
                L_plus_U[i][j] = 1.0;
            }
            else if (j == i-1) {
                A[i][j] = -1.0;
                L_plus_U[i][j] = 1.0;
            } 
            else {
                A[i][j] = 0.0;
                D_inverted[i][j] = 0.0;
                L_plus_U[i][j] = 0.0;
            }
        }
    }


    
    
    //Set values for vector b
    for (i = 0; i < N-1; i++) {
        b[i] = 0;
    }
    b[N-1] = N + 1;
    
    
    /* STAGE: 3 , JACOBI process */
    
    
    printf("Number of threads: %d\n", omp_get_max_threads());

    
    //We initialize the x_old = 1.0, because in the first step we want to multiply with #0  
    for (i = 0; i < N; i++) {
        x_old[i] = 1.0;
    }
    
    //Time starts
    start = omp_get_wtime();
    
    for (i = 1; i <= M; i++) { //i = iterations
        residual = 0;
        difference = 0;
        
        #pragma omp parallel shared(A, D_inverted, L_plus_U, x_old, x_new, b, x_temp) 
        {
            
            //We calculate the x_new
            #pragma omp for private(k)
                for (j = 0; j < N; j++) {
                    x_new[j] = 0.0;
                    for (k = 0; k < N; k++) {
                        x_new[j] += L_plus_U[j][k] * x_old[k];
                    }
                    x_new[j] += b[j];
                    x_new[j] = D_inverted[j][j] * x_new[j];
                
                }
            
            //Calculating residual, difference
            #pragma omp for private(k)
                for (j = 0; j < N; j++) {
                    x_temp[j] = 0.0;
                    for (k = 0; k < N; k++) {
                        x_temp[j] += A[j][k] * x_new[k];
                    }
                        x_temp[j] = pow(b[j] - x_temp[j], 2);
                    
                    #pragma omp critical (residual)
                        { 
                            residual += x_temp[j];
                        }

                    x_temp[j] = pow(x_old[j]-x_new[j], 2);
                    #pragma omp critical (difference)
                        { 
                            difference += x_temp[j];
                        }
                    }
            #pragma omp for
                //x_old gets the value of x_new we calculated before
                for (j = 0; j < N; j++) {
                    x_old[j] = x_new[j];
                }
        }
        printf("iter = %d, residual = %.2f, difference = %.2f\n",i, sqrt(residual), sqrt(difference));
    }
    end = omp_get_wtime();
    fprintf(stderr, "Execution time = %lf sec\n", end-start);
    
    
    /*STAGE: 4 */
    //end phase: free
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(L_plus_U[i]);
		free(D_inverted[i]);
		
    }
    free(A);
    free(L_plus_U);
	free(D_inverted);
	free(x_temp);
    free(b);
    free(x_old);
    free(x_new);
    
    
    return 0;
}
