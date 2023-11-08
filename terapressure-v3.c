/*********************************************************
* Terapressure v3
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
*
 * Changelog:
* v1: MPI_Send and MPI_Recv on every border cell
* v2: MPI_Send and MPI_Recv to send/recv border rows/columns
*     it is fasters, less mpi messages
* v3: Memory allocation in 1D vector, less copies
*     Extended size blocks, less loops
*     Fastest version
*
 **********************************************************/
 
#define VERSION "v3"
 
#include <stdio.h>
#include <stdlib.h> // malloc, exit
#include <mpi.h>
 
int main(int argc, char *argv[])
{
  int N=20, M=30;      // number of cells NxM
  int n=2,  m=3;        // number of blocks nxm
  int tpi=16, tpj=18;   // test pressure coordinates
  int tai=7, taj=9;     // test average coordinates
  int i, j, I, J;       // local and global i,j
  int myi, myj;         // my i,j in neighbor map
  int by, bx;           // block size in y and x direction
  int numprocs, myid;   // number of processors and my rank id
  double *P, *A;        // 2D array of pressures and averages
                        // (1D continues array for performance)
 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
 
  // get command line arguments if any
  if (argc > 1) {
    if (argc != 5) {
      if (myid==0) {
        fprintf(stderr, "usage: prog [N M n m]\n");
        fprintf(stderr, "Parameters:\n");
        fprintf(stderr,
          "\tN: number of rows or cells in y direction. Default: %ld\n", N);
        fprintf(stderr,
          "\tM: number of columns or cells in x direction. Default: %ld\n", M);
        fprintf(stderr,
          "\tn: number of blocks in y direction. Default: %d\n", n);
        fprintf(stderr,
          "\tm: number of blocks in x direction. Default %d\n", m);
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
  int width = bx + 2;
  int height = by + 2;
 
  // start message
  if (myid==0) {
    printf("===============\n");
    printf("Terapressure %s\n", VERSION);
    printf("===============\n");
    printf("Number of cells: %lu (%lu x %lu)\n", N*M, N, M);
    printf("Number of blocks: %d (%d x %d)\n", n*m, n, m);
    printf("Number of processors %d\n", numprocs);
    printf("Block size: (%d x %d)\n", by, bx);
    printf("Block size extended (height,width): (%d x %d)\n", height, width);
    printf("Aprox. memory needed per processor: %.2f GB\n",
      height*width*2*sizeof(double*)/1024.0/1024.0/1024.0);
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
      fprintf(stderr,
        "Number of processors must be the same as number of blocks: %d\n",
        n*m);
    MPI_Finalize();
    exit(2);
  }
 
  double t = MPI_Wtime();
 
  // memory allocation
  P = malloc(sizeof(double*) * width * height);
  A = malloc(sizeof(double*) * width * height);
  int B[n][m];
 
  // initialize
  for (i=0; i < width * height; i++) {
    P[i] = 0;
    A[i] = 0;
  }
 
 
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
 
  // show neighbors map
  if (myid==0) {
    printf("Neighbors map:\n");
    for (i=0; i < n; i++) {
      for (j=0; j < m; j++) {
        printf ("%3d ",  B[i][j]);
      }
      printf ("\n");
    }
  }
 
  // compute pressures
  double pressure = -1;
 
  for (i=1; i < by+1; i++) {
    I = myi * by + i - 1;
    for (j=1; j < bx+1; j++) {
      J = myj * bx + j - 1;
      if (I==0 || I==N-1 || J==0 || J==M-1)
        P[i*width+j] = 0;
      else
        P[i * width + j] = (double)(I+J) * (double)(I*J);
      if (I == tpi && J == tpj)
        pressure = P[i * width + j];
    }
  }
 
  // print pressures 
  if (myid==4) {
    printf("%d: Pressures:\n", myid);
    for (i=0; i < width * height; i++) {
        printf("%5.0f", P[i]);
        if ((i+1) % width == 0)
          printf("\n");
    }
    printf("\n");
  }
 
 
  // average pressure
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
    b = B[myi-1][myj];
    if ((myi+myj) % 2 == 0) {
      // printf("%2d: send %d top row\n", myid, b);
      MPI_Send(&P[width+1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
      // printf("%2d: recv %d bottom row\n", myid, b);
      MPI_Recv(&P[0 * width + 1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      // printf("%2d: recv %d bottom row\n", myid, b);
      MPI_Recv(&P[0*width+1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // printf("%2d: send %d top row\n", myid, b);
      MPI_Send(&P[width+1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
    }
  }
  // bottom row
  if (Ie < N-1) {
    b = B[myi+1][myj];
    if ((myi+myj) % 2 == 0) {
      // printf("%2d: send %d bottom row\n", myid, b);
      MPI_Send(&P[by*width+1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
      // printf("%2d: recv %d top row\n", myid, b);
      MPI_Recv(&P[(by+1)*width+1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      // printf("%2d: recv %d top row\n", myid, b);
      MPI_Recv(&P[(by+1)*width+1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // printf("%2d: send %d bottom row\n", myid, b);
      MPI_Send(&P[by*width+1], bx, MPI_DOUBLE, b, 0, MPI_COMM_WORLD);
    }
  }
  // left column
  if (Js > 0) {
    for (i=1; i < by+1; i++)
      send_y[i-1] = P[i * width + 1];
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
      P[i * width + 0] = recv_y[i-1];
  }
  // right column
  if (Je < M-1) {
    for (i=1; i < by+1; i++)
      send_y[i-1] = P[i * width + bx];
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
      P[i * width + (bx+1)] = recv_y[i-1];
  }
 
  // print extended presures
  /*
  if (myid==0) {
    printf("%d: Extended pressures:\n", myid);
      for (i=0; i < width * height; i++) {
        printf("%5.0f", P[i]);
        if ((i+1) % width == 0)
          printf("\n");
     }
     printf("\n");
  }
  */
 
  double average = -1;
 
  for (i=1; i < by+1; i++) {
    I = myi * by + i - 1;
    for (j=1; j < bx+1; j++) {
      J = myj * bx + j - 1;
      if ( I==0 || I==N-1 || J==0 || J==M-1 )
        continue;
      //A[i][j] = ( P[i][j] + P[i-1][j] + P[i+1][j] + P[i][j-1] + P[i][j+1] ) / 5;
      A[i * width + j] = (
          P[i*width+j]
          + P[(i-1)*width+j]
          + P[(i+1)*width+j]
          + P[i*width+(j-1)]
          + P[i*width+(j+1)]
        ) / 5;
 
      if (I==tai && J==taj)
        average = A[i * width + j] ;
    }
  }
 
  // test
  /*
  int ok = 1;
  double expected_avg = 0.0;
 
  for (i=0; i < by+2; i++) {
    I = myi * by + i - 1;
    for (j=0; j < bx+2; j++) {
      J = myj * bx + j - 1;
      if ( I<=0 || I>=N-1 || J<=0 || J>=M-1 )
        continue;
      P[i * width + j] = (double)(I+J) * (double)(I*J);
    }
  }

  if (myid==0) {
    printf("%d: Extended pressures:\n", myid);
      for (i=0; i < width * height; i++) {
        printf("%5.0f", P[i]);
        if ((i+1) % width == 0)
          printf("\n");
     }
     printf("\n");
  }

  for (i=1; i < by+1; i++) {
    I = myi * by + i - 1;
    for (j=1; j < bx+1; j++) {
      J = myj * bx + j - 1;
      if ( I==0 || I==N-1 || J==0 || J==M-1 )
        continue;
      expected_avg = A[i * width + j];
      A[i * width + j] = ( P[i*width+j] + P[(i-1)*width+j] + P[(i+1)*width+j] + P[i*width+(j-1)] + P[i*width+(j+1)] ) / 5;
      if (myid==0 && expected_avg != A[i * width + j]) {
        printf("%d: Average incorrect at (%d,%d): expected: %.2f, actual: %.2f\n",
                 myid,I,J,expected_avg,A[i * width + j]);
        ok = 0;
        break;
      }
    }
    if (!ok)
      break;
  }
  */
 
  // print averages
  /*
  if (myid==0) {
    printf("%d: Averages:\n", myid);
    for (i=0; i < width * height; i++) {
        printf("%8.2f", A[i]);
        if ((i+1) % width == 0)
          printf("\n");
    }
    printf("\n");
  }
  */
 
  // cleanup memory
  free(P);
  free(A);
 
  // report result
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
