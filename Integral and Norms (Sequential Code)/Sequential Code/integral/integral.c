
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

// This function calculates the area under a function using the trapezoidal
// rule.

// parallel specific variables
int thread_count;
double GLOBAL_MUTEX_SUM = 0;
pthread_mutex_t mutex;


// trapezoid globals
double a;         // left endpoint
double b;         // right endpoint
long int n;       // number of trapezoids
double h;         // height of trapezoids

double f(double x);
void* mutex_trapezoid(void* rank);

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n)  (BLOCK_LOW((id)+1,p,n)-1)

int main( int argc, char **argv ) {

  printf("Enter a (left endpt), b (right endpt), n (# trapezoids), p (threads counts)\n");
  scanf("%lf", &a);
  scanf("%lf", &b);
  scanf("%ld", &n);
  scanf("%d", &thread_count);
  h = (b-a)/n;	

  pthread_t* thread_handles;

  GLOBAL_MUTEX_SUM = 0;

  // create thread handles and initialize mutex
  pthread_mutex_init(&mutex, NULL);
  thread_handles = malloc( thread_count * sizeof(pthread_t));

  // create the pthreads on mutex trapezoid function
  long thread;
  for( thread = 0; thread < thread_count; thread++)
    pthread_create( &thread_handles[thread], NULL, mutex_trapezoid, 
      (void*) thread );

  // join all the thread handles
  for( thread=0; thread < thread_count; thread++)
    pthread_join( thread_handles[thread], NULL);

  // free thread handles and mutex
  free(thread_handles);
  pthread_mutex_destroy(&mutex);
 
  printf("----- %d threads -----\n", thread_count);
  printf("With n = %ld trapezoids, our estimate\n", n);
  printf("of the integral from %f to %f = %.15f\n", a, b, GLOBAL_MUTEX_SUM);  

  return 0;
}

void* mutex_trapezoid(void* rank)
{ 
  long thread_rank = (long)rank;
  double local_sum = 0.0;
  long long i;
  int special_case = (int)thread_rank;

  // allocate a chunk of work to the thread
  long long my_first_i = BLOCK_LOW(thread_rank, thread_count, n);
  long long my_last_i = BLOCK_HIGH(thread_rank, thread_count, n);

  // let thread with rank 1 add (f(a)+f(b))/2.0 to it's sum. This is only
  // done once for trapezoidal rule calculation & thread that does it is
  // should be 1st or 2nd in case code is only run on 2 cores.
  if( special_case == 1)
    local_sum += (f(a)+f(b))/2.0;

  for( i= my_first_i; i <= my_last_i; i++)
    local_sum += f(a+(i*h));
  
  local_sum *= h;

  // update the global sum
  pthread_mutex_lock(&mutex);
  GLOBAL_MUTEX_SUM += local_sum;
  pthread_mutex_unlock(&mutex);

  return NULL;
}

double f(double x) {
  return x*x;
}