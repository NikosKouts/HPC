#ifndef _HEADER_
#define _HEADER

#include <stdio.h>
#include <stdlib.h>
 #include <time.h>
 #include <string.h>

//#define DEBUG
#define NUMBER 1


double **array_constructor(int n, double number);
double first_norm(double **array, int n, int k);
double infinite_norm(double **array, int n, int k);



#endif