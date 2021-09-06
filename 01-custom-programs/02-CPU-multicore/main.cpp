#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <omp.h>

void matmul(double *a, double *b, double *c, long n){
    #pragma omp parallel for
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            double acc = 0;
            long index = i*n + j;
            for(int k=0; k<n; ++k){
                acc += a[i*n + k]*b[k*n + i];
            }
            c[index] = acc; 
        }
    }
}

void init(double *a, long n, double val){
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n + j;
            a[index] = val; 
        }
    }
}

void printMat(double *a, long n, const char *msg){
    printf("%s:\n", msg);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            long index = i*n + j;
            printf("%.2f ", a[index]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv){
    double t1, t2;
	if(argc != 3){
		fprintf(stderr, "run as ./prog n nt\n");
        exit(EXIT_FAILURE);
    }
    long n = atoi(argv[1]);
    int nt = atoi(argv[2]);
	printf("matmul cpu %lu x %lu\n", n, n);
    printf("using %i threads\n", nt);
    omp_set_num_threads(nt);
	printf("Mallocs.............."); fflush(stdout);
	double* matA = new double[n*n];
	double* matB = new double[n*n];
	double* matC = new double[n*n];
	printf("done\n"); fflush(stdout);
	printf("Init Matrices........"); fflush(stdout);
    init(matA, n, 1.0); 
    init(matB, n, 2.0); 
	printf("done\n"); fflush(stdout);
	printf("Matmul..............."); fflush(stdout);
    t1 = omp_get_wtime();
    matmul(matA, matB, matC, n);
    t2 = omp_get_wtime();
	printf("done: %f secs\n", t2-t1); fflush(stdout);
    if(n <= 32){
        printMat(matC, n, "C Mat");
    }
	exit(EXIT_SUCCESS);
}
