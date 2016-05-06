/*********************************************************
 * Terapressure 0.1
 * Exercise from Majdi MPI course for SSD on 05/20/2016
 * Author: Santiago
 * Date: That weekend.
 * 
 * MPI implementation to calculate pressures using:
 *     P[i][j] = (double)(I+J) * (double)(I*J);
 * and average pressures using:
 *     A[i][j] = (center + left + top + right + bottom) / 5;
 * in a 2D array of arbitrary number of cells.
 * This is just a fake formula to implement MPI communication
 * and understand the parallel paradigm in reservoir simulation.
 * The reason for the name is the support of TB cells :).
 *
 * Compile: mpicc terapressure.c -o terapressure
 * Run:    mpirun -np 6 ./terapressure
 * or:     mpirun -np 64 --host a,b ./terapressure 1024 1024 8 8
 *
 * Output:
 * 
 * Terapressure v0.1
 * =================
 * Number of cells: 600 (20 x 30)
 * Number of blocks: 6 (2 x 3)
 * Number of processors 6
 * Block size: (10 x 10)
 * Preasure at (16,18): 9792.00 computed by processor 4
 * Average  at ( 7, 9): 1014.40 computed by processor 0
 * Time elapsed: 0.01 seconds.
 **********************************************************/

#include <stdio.h>
#include <stdlib.h> // malloc, exit
#include "mpi.h"

int main(int argc, char *argv[])
{
  long N=20, M=30;      // number of cells NxM
  int n=2,  m=3;        // number of blocks nxm 
  int tpi=16, tpj=18;   // test pressure coordinates
  int tai=7, taj=9;     // test average coordinates
  int i, j, I, J;       // local and global i,j
  int myi, myj;         // my i,j in neighbor map
  int bi, bj;           // block size in y and x direction
  int numprocs, myid;   // number of processors and my rank id
  double **P, **A;      // 2D array of pressures and averages
  int **B;              // 2D array with map of neighbors

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // get command line arguments if any
  if (argc > 1) {
    if (argc != 5) {
      if (myid==0) {
        fprintf(stderr, "usage: prog [N M n m]\n");
        fprintf(stderr, "Parameters:\n");
        fprintf(stderr, "\tN: number of rows or cells in y direction. Default: %ld\n", N);
        fprintf(stderr, "\tM: number of columns or cells in x direction. Default: %ld\n", M);
        fprintf(stderr, "\tn: number of blocks in y direction. Default: %d\n", n);
        fprintf(stderr, "\tm: number of blocks in x direction. Default %d\n", m);
      }
      MPI_Finalize();
      exit(3);
    } 
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    n = atoi(argv[3]);
    m = atoi(argv[4]);
  }

  bi = N/n;
  bj = M/m;

  // start message
  if (myid==0) {
    printf("Terapressure v0.1\n");
    printf("=================\n");
    printf("Number of cells: %lu (%lu x %lu)\n", N*M, N, M);
    printf("Number of blocks: %d (%d x %d)\n", n*m, n, m);
    printf("Number of processors %d\n", numprocs);
    printf("Block size: (%d x %d)\n", bi, bj);
  }
  
  // validate parameters   
  if (N % n != 0 || M % m != 0) {
    if(myid==0) 
      fprintf(stderr,"Number of blocks in x or y axis do not fit.\n"); 
    MPI_Finalize();
    exit(1);
  }  
  if (numprocs != n*m) {
    if (myid==0) 
      fprintf(stderr,"Number of processors must be the same as number of blocks: %d\n", n*m);
    MPI_Finalize();
    exit(2);
  }

  double t = MPI_Wtime();

  // memory allocation
  // stack allocation is simple but limited in size
  // double   P[bi][bj];
  // double   A[bi][bj];
  // int     B[n][m];      
  
  // heap allocation
  P = malloc(sizeof(double*) * bi); 
  A = malloc(sizeof(double*) * bi); 
  for (i=0; i < bi; i++) {
    P[i] = malloc(sizeof(double) * bj);
    A[i] = malloc(sizeof(double) * bj);
  }  
  B = malloc(sizeof(int*) * n); 
  for (i=0; i < n; i++) {
    B[i] = malloc(sizeof(int) * m);
  }
  
  // domain decomposition
  int rank = 0;    
  //printf("Neighbors map:\n");
  for (i=0; i < n; i++) {
    for (j=0; j < m; j++) {
      if (rank == myid) {
        myi = i; 
        myj = j;
      }
      B[i][j] = rank++;
      //printf ("%3d ",  W[i][j]);
    }
    //printf ("\n");
  }
  //printf("%d: my i,j in neighbor map: %d,%d\n", myid, myi, myj);

  // compute pressures
  // printf("%d: My pressures:\n", myid);
  double pressure = -1;

  for (i=0; i < bi; i++) {
    I = myi * bi + i;
    for (j=0; j < bj; j++) {
      J = myj * bj + j;
      if (I==0 || I==N-1 || J==0 || J==M-1)
        P[i][j] = 0;
      else
        P[i][j] = (double)(I+J) * (double)(I*J);
      //printf ("L(%d,%d) G(%d,%d): %.2f\t",i,j,I,J, P[i][j]);
      if (I == tpi && J == tpj) 
        pressure = P[i][j];
    }
    //printf ("\n");
  }

  // average pressure
  int neighbor;
  double center, left, top, right, bottom;  
  double average = -1;

  for (i=0; i < bi; i++) {
    I = myi * bi + i;
    for (j=0; j < bj; j++) {
      J = myj * bj + j;
      if ( I==0 || I==N-1 || J==0 || J==M-1 )
        continue;
    
      center = P[i][j];
      
      // top cell
      if (i==0) {
        neighbor = B[myi-1][myj]; 
        MPI_Send(&center, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD);
        MPI_Recv(&top, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);      
        //printf("%2d: send to   %d (%d,%d): %.2f\n", myid, neighbor,I,J,center);
        //printf("%2d: recv from %d (%d,%d): %.2f\n", myid, neighbor,I-1,J,top);
      } else {
        top = P[i-1][j];
      }
      // bottom cell
      if (i==bi-1) {
        neighbor = B[myi+1][myj];
        MPI_Send(&center, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD);
        MPI_Recv(&bottom, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //printf("%2d: send to   %d (%d,%d): %.2f\n", myid, neighbor,I,J,center);
        //printf("%2d: recv from %d (%d,%d): %.2f\n", myid, neighbor,I+1,J,bottom);
      } else {
        bottom = P[i+1][j];
      }
      // left cell
      if (j==0) {
        neighbor = B[myi][myj-1];
        MPI_Send(&center, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD);
        MPI_Recv(&left, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //printf("%2d: send to   %d (%d,%d): %.2f\n", myid, neighbor,I,J,center);
        //printf("%2d: recv from %d (%d,%d): %.2f\n", myid, neighbor,I,J-1,left);
      } else {
        left = P[i][j-1];
      }
      // right cell
      if (j==bj-1) {
        neighbor = B[myi][myj+1];
        MPI_Send(&center, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD);
        MPI_Recv(&right, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //printf("%2d: send to   %d (%d,%d): %.2f\n", myid, neighbor,I,J,center);
        //printf("%2d: recv from %d (%d,%d): %.2f\n", myid, neighbor,I,J+1,right);
      } else {
        right = P[i][j+1];
      }
  
      A[i][j] = ( center + left + top + right + bottom ) / 5;    
      
      //printf ("L(%d,%d) G(%d,%d): %.2f\t",i,j,I,J, A[i][j]);       
      
      if (I==tai && J==taj) 
        average = A[i][j];
    }
    //printf ("\n");
  }

  // cleanup memory  
  for (i=0; i < bi; i++) {
    free(P[i]);
    free(A[i]);
  }
  free(P); 
  free(A);
  for (i=0; i < n; i++) {
    free(B[i]);
  }
  free(B);

  // report result
  //printf("Preasure at (16,18): %.2f\n", P[16][18]);
  //printf("Avg at (7,9): %.2f\n", A[7][9]);
 
  if (pressure > -1) 
    printf("Preasure at (%2d,%2d): %.2f computed by processor %d\n",
      tpi, tpj, pressure, myid);
  if (average > -1)
    printf("Average  at (%2d,%2d): %.2f computed by processor %d\n",
      tai, taj, average, myid);

  MPI_Barrier(MPI_COMM_WORLD); 
  if (myid==0)
    printf("Time elapsed: %.2f seconds.\n", MPI_Wtime()-t);
 
  MPI_Finalize();

  return 0;
}


