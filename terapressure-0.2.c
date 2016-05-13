/*********************************************************
* Terapressure 0.2
* Exercise MPI course on 05/20/2016
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
* Output:
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

#define debug 1

int main(int argc, char *argv[])
{
  long N=8, M=8;      // number of cells NxM
  int n=2,  m=2;        // number of blocks nxm
  int tpi=3, tpj=3;   // test pressure coordinates
  int tai=3, taj=3;     // test average coordinates
  int i, j, I, J;       // local and global i,j
  int myi, myj;         // my i,j in neighbor map
  int by, bx;           // block size in y and x direction
  int numprocs, myid;   // number of processors and my rank id
  double **P, **A;      // 2D array of pressures and averages

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

  by = N/n;
  bx = M/m;

  // start message
  if (myid==0) {
    printf("Terapressure v0.2\n");
    printf("=================\n");
    printf("Number of cells: %lu (%lu x %lu)\n", N*M, N, M);
    printf("Number of blocks: %d (%d x %d)\n", n*m, n, m);
    printf("Number of processors %d\n", numprocs);
    printf("Block size: (%d x %d)\n", by, bx);
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
  P = malloc(sizeof(double*) * (by + 2));
  for (i=0; i < by+2; i++)
    P[i] = malloc(sizeof(double) * (bx + 2));

  A = malloc(sizeof(double*) * (by + 2));
  for (i=0; i < by+2; i++)
    A[i] = malloc(sizeof(double) * (bx + 2));

  int B[n][m];

  // initialize
  for (i=0; i < by+2; i++)
    for (j=0; j < bx+2; j++)
      P[i][j] = 0;

  // domain decomposition
  int rank = 0;
  for (i=0; i < n; i++) {
    for (j=0; j < m; j++) {
      if (rank == myid) {
        myi = i;
        myj = j;
      }
      B[i][j] = rank++;
    }
  }

  // compute pressures
  double pressure = -1;

  for (i=1; i < by+1; i++) {
    I = myi * by + i - 1;
    for (j=1; j < bx+1; j++) {
      J = myj * bx + j - 1;
      if (I==0 || I==N-1 || J==0 || J==M-1)
        P[i][j] = 0;
      else
        P[i][j] = (double)(I+J) * (double)(I*J);
      if (I == tpi && J == tpj)
        pressure = P[i][j];
    }
  }


  // if (myid==0) {
  //   for (i=0; i < by+2; i++) {
  //     for (j=0; j < bx+2; j++)
  //       printf("%5.0f",P[i][j]);
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // average pressure
  double send_x[bx];
  double recv_x[bx];
  double send_y[by];
  double recv_y[by];

  // global I start, I end, J start, J end
  int Is = myi * by;
  int Ie = Is + by - 1;
  int Js = myj * bx;
  int Je = Js + bx - 1;
  int b; // neighbor

  // top row
  if (Is > 0) {
    for (j=1; j < bx+1; j++)
      send_x[j-1] = P[1][j];
    b = B[myi-1][myj];
    if ((myi+myj) % 2 == 0) {
      // printf("%2d: send %d top row\n", myid, b);
      MPI_Send(send_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
      // printf("%2d: recv %d bottom row\n", myid, b);
      MPI_Recv(recv_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      // printf("%2d: recv %d bottom row\n", myid, b);
      MPI_Recv(recv_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // printf("%2d: send %d top row\n", myid, b);
      MPI_Send(send_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
    }
    for (j=1; j < bx+1; j++)
      P[0][j] = recv_x[j-1];
  }
  // bottom row
  if (Ie < N-1) {
    for (j=1; j < bx+1; j++)
      send_x[j-1] = P[bx][j];
    b = B[myi+1][myj];
    if ((myi+myj) % 2 == 0) {
      // printf("%2d: send %d bottom row\n", myid, b);
      MPI_Send(send_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
      // printf("%2d: recv %d top row\n", myid, b);
      MPI_Recv(recv_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      // printf("%2d: recv %d top row\n", myid, b);
      MPI_Recv(recv_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // printf("%2d: send %d bottom row\n", myid, b);
      MPI_Send(send_x, bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
    }
    for (j=1; j < bx+1; j++)
      P[bx+1][j] = recv_x[j-1];
  }
  // left column
  if (Js > 0) {
    for (i=1; i < by+1; i++)
      send_y[i-1] = P[i][1];
    b = B[myi][myj-1];
    if ((myi+myj) % 2 == 0) {
      // printf("%2d: send %d left column\n", myid, b);
      MPI_Send(send_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
      // printf("%2d: recv %d right column\n", myid, b);
      MPI_Recv(recv_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      // printf("%2d: recv %d right column\n", myid, b);
      MPI_Recv(recv_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // printf("%2d: send %d left column\n", myid, b);
      MPI_Send(send_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
    }
    for (i=1; i < by+1; i++)
      P[i][0] = recv_y[i-1];
  }
  // right column
  if (Je < M-1) {
    for (i=1; i < by+1; i++)
      send_y[i-1] = P[i][by];
    b = B[myi][myj+1];
    if ((myi+myj) % 2 == 0) {
      // printf("%2d: send %d right column\n", myid, b);
      MPI_Send(send_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
      // printf("%2d: recv %d left column\n", myid, b);
      MPI_Recv(recv_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      // printf("%2d: recv %d left column\n", myid, b);
      MPI_Recv(recv_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // printf("%2d: send %d right column\n", myid, b);
      MPI_Send(send_y, by, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
    }
    for (i=1; i < by+1; i++)
      P[i][by+1] = recv_y[i-1];
  }

  // if (myid==2) {
  //   for (i=0; i < by+2; i++) {
  //     for (j=0; j < bx+2; j++) {
  //       printf("%5.0f",P[i][j]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  double average = -1;

  for (i=1; i < by+1; i++) {
    I = myi * by + i - 1;
    for (j=1; j < bx+1; j++) {
      J = myj * bx + j - 1;
      if ( I==0 || I==N-1 || J==0 || J==M-1 )
        continue;
      A[i][j] = ( P[i][j] + P[i-1][j] + P[i+1][j] + P[i][j-1] + P[i][j+1] ) / 5;
      if (I==tai && J==taj)
        average = A[i][j];
    }
  }

  // cleanup memory
  for (i=0; i < by+2; i++) {
    free(P[i]);
    free(A[i]);
  }
  free(P);
  free(A);


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

