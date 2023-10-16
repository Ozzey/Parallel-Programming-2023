#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//int const n = 10e8;

double f(double x) {
    return (4 / (1 + x * x));
}

double partialArea(double indexSegment, int n, double h) {
    double area = 0.0;
    for (int i = 0; i < n; i++) {
        area += (0.5 * h * (f(indexSegment * h + i * h) + f(indexSegment * h + (i + 1) * h)));
    }
    return area;
}

int main(int argc, char *argv[]) {
    int myrank, size;
    MPI_Status Status; // MPI data type
    double integral = 0.0;
    double startwtime, endwtime;
    double a = 0.0, b = 1.0;
    int n;

    if (argc != 2) {
   	printf("Usage: %s <n>\n", argv[0]);
        exit(1);
   }

    n = atoi(argv[1]); // Get n from command line argument
    double h = (b - a) / n;

    /* MPI programs start with MIP_Init; all "N" processes exist thereafter */
    MPI_Init(&argc, &argv);

    /* find out how big the world of processes is */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* and this process's rank is */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MPI_Barrier(MPI_COMM_WORLD);
    startwtime = MPI_Wtime();
    if (myrank == 0) {
        for (int i = 0; i < n; i++) {
            integral += (0.5 * h * (f(a + i * h) + f(a + (i + 1) * h)));
        }
        printf("----------------------------------------------------\n");
        printf("Integral by sequential: %lf\n", integral);
        printf("Size (number of processes): %d\n", size);
        printf("N (number of small segments): %d\n", n);
        printf("----------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    endwtime = MPI_Wtime();
    double sequential_time = endwtime - startwtime;

    MPI_Barrier(MPI_COMM_WORLD);
    startwtime = MPI_Wtime();

    int k = n / size; // number of small intervals 1 process has to work with

    int remain = n % size;
    int for_main = k + remain; // if there is a remainder n/size, then it is for the main process
    double result = 0.0;
    double partial = 0.0;
    int index;

    if (myrank == 0) {
        result += partialArea(0, for_main, h);
        for (int i = 1; i < size; i++) {
            MPI_Recv(&partial, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result += partial;
        }
        printf("----------------------------------------------------\n");
        printf("Integral by the addition of all parts: %lf\n", result);
        printf("----------------------------------------------------\n");
    } else {
        index = for_main + k * (myrank - 1);
        partial = partialArea(index, k, h);
        MPI_Send(&partial, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        printf("Rank %d: Partial Integral I%d = %lf\n", myrank, myrank, partial);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    endwtime = MPI_Wtime();
    double process_time = endwtime - startwtime;

    if (myrank == 0) {
        printf("----------------------------------------------------\n");
        printf("Sequential time: %lf\n", sequential_time);
        printf("Process time: %lf\n", process_time);
        printf("Speed up: %lf\n", sequential_time / process_time);
        printf("----------------------------------------------------\n");
    }
    MPI_Finalize();
}
