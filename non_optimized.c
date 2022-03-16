#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>

#define N 1024


int main(void) {

    int iteration = 0;
    double max_err = 1.0;
    double delta = 10.0 / (N - 1);

    double** U = (double**)calloc(N, sizeof(double*));
    double** U_n = (double**)calloc(N, sizeof(double*));

    for (int i = 0; i < N; i++) {
        U[i] = (double*)calloc(N, sizeof(double));
        U_n[i] = (double*)calloc(N, sizeof(double));
    }

    U[0][0] = 10.0;
    U[0][N - 1] = 20.0;
    U[N - 1][0] = 20.0;
    U[N - 1][N - 1] = 30.0;

    for (int i = 0; i < N; i++) {
        U[i][0] = 10 + delta * i;
        U[0][i] = 10 + delta * i;
        U[N - 1][i] = 20 + delta * i;
        U[i][N - 1] = 20 + delta * i;

        U_n[i][0] = U[i][0];
        U_n[0][i] = U[0][i];
        U_n[N - 1][i] = U[N - 1][i];
        U_n[i][N - 1] = U[i][N - 1];
    }


#pragma acc data copyin(U[0:N][0:N], U_n[0:N][0:N])
    {
        while (max_err > 1e-6 && iteration < 1e+6) {

            iteration++;
            max_err = 0.0;

#pragma acc kernels 
            {
#pragma acc loop independent collapse(2) reduction(max:max_err)  
                for (int i = 1; i < N - 1; i++)
                    for (int j = 1; j < N - 1; j++) {
                        U_n[i][j] = 0.25 * (U[i + 1][j] + U[i - 1][j] + U[i][j - 1] + U[i][j + 1]);
                        max_err = fmax(max_err, U_n[i][j] - U[i][j]);
                    }
            }

#pragma acc kernels
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    U[i][j] = U_n[i][j];
            if (iteration % 100 == 0)
                printf("%d %lf\n", iteration, max_err);
        }
    }

    printf("%d %lf\n\n", iteration, max_err);

    return 0;
}