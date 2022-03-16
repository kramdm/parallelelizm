#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#include <math.h>

#define N 128

int main(void)
{
    double** copy_pointer;
    int iteration = 0;
    double max_err = 1.0;
    double delta = 10.0 / (N - 1);

    double** U = (double**)calloc(N, sizeof(double*));
    double** U_n = (double**)calloc(N, sizeof(double*));

    for (int i = 0; i < N; i++) {
        U[i] = (double*)calloc(N, sizeof(double));
        U_n[i] = (double*)calloc(N, sizeof(double));
    }

#pragma acc enter data create(U[0:N][0:N], U_n[0:N][0:N]) copyin(N, delta)
    

#pragma acc kernels
    

    {
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
    }


#pragma acc data create(max_err)
    {
        while (max_err > 1e-6 && iteration < 10) {

            iteration++;

            if (iteration % 100 == 0) {
#pragma acc kernels 
                max_err = 0.0;

#pragma acc data present(U, U_n)
                

#pragma acc kernels async(1)


                {
#pragma acc loop independent collapse(2) reduction(max:max_err)
                    

                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++) {
                            U_n[i][j] = 0.25 * (U[i + 1][j] + U[i - 1][j] + U[i][j - 1] + U[i][j + 1]);
                            max_err = fmax(max_err, U_n[i][j] - U[i][j]);
                        }
                }

            }
            else {

#pragma acc data present(U, U_n)
#pragma acc kernels async(1)
                {
#pragma acc loop independent collapse(2)
                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++)
                            U_n[i][j] = 0.25 * (U[i + 1][j] + U[i - 1][j] + U[i][j - 1] + U[i][j + 1]);
                }
            }

            copy_pointer = U;
            U = U_n;
            U_n = copy_pointer;

            if (iteration % 100 == 0) {
#pragma acc wait(1)

#pragma acc update host(max_err)
    

                printf("%d %lf\n", iteration, max_err);
            }
        }
    }
    printf("%d %lf\n", iteration, max_err);


    return 0;
}


