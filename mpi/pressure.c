/*********************************************************
 * Calculate pressures and averages in a 2D grid
 * Exercise MPI course on 05/20/2016
 * Author: Santiago
 * Serial version
 * Compile: gcc -o pressure pressure.c
 * Run: ./pressure
 * Output:
 * Preasure at (16,18): 9792.00
 * Average  at ( 7, 9): 1014.40
 *********************************************************/

#include <stdio.h>

int main(int argc, char *argv[])
{
  int N = 8, M = 8;     // number of cells N by M
  int tpi=3, tpj=3;     // test pressure coordinates
  int tai=3, taj=3;       // test average coordinates
  int i, j;    		    
  double P[N][M];
  double A[N][M];

  // initialize with zeros
  for (i=0; i < N; i++)
    for (j=0; j < M; j++)  
      P[i][j] = 0;
  
  // compute pressures
  for (i=1; i < N-1; i++)
    for (j=1; j < M-1; j++)  
      P[i][j] = (double)(i+j) * (double)(i*j);

  // compute average pressures
  for (i=1; i < N-1; i++)
    for (j=1; j < M-1; j++) 
      A[i][j] = ( P[i][j] + P[i-1][j] + P[i+1][j] + P[i][j-1] + P[i][j+1] ) / 5;		

  printf("Preasure at (%2d,%2d): %.2f\n", tpi, tpj, P[tpi][tpj]);
  printf("Average  at (%2d,%2d): %.2f\n", tai, taj, A[tai][taj]);

  return 0;
}


