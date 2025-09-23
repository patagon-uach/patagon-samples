#pragma once
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include <cuda_runtime.h>
#include <cstdio>

// PRINT GPU INFO
static void print_gpu_specs(int dev){
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Memory:                       %.3f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %d\n", prop.concurrentKernels);

    int memClockKHz = 0;
    int busWidthBits = 0;
    cudaDeviceGetAttribute(&memClockKHz,  cudaDevAttrMemoryClockRate,        dev); // kHz
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth,   dev); // bits

    if (memClockKHz > 0 && busWidthBits > 0) {
        double memClockMHz = memClockKHz / 1000.0;
        double peakGBs     = 2.0 * (memClockKHz / 1e6) * (busWidthBits / 8.0); // DDR assumption
        printf("  Memory Clock Rate:            %.0f MHz\n", memClockMHz);
        printf("  Memory Bus Width:             %d bits\n", busWidthBits);
        printf("  Peak Memory Bandwidth:        %.3f GB/s\n\n", peakGBs);
    } else {
        printf("  Memory Clock Rate:            N/A\n");
        printf("  Memory Bus Width:             N/A\n");
        printf("  Peak Memory Bandwidth:        N/A\n\n");
    }
}



// CUBLAS ALGORITHMS
#define NUM_CUBLAS_ALGS 2
cublasGemmAlgo_t cublasAlgs[NUM_CUBLAS_ALGS] = {CUBLAS_GEMM_DEFAULT, 
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP};

// CUBLAS MATH MODES
#define NUM_CUBLAS_MATH_MODES 4
cublasMath_t cublasMathModes[NUM_CUBLAS_MATH_MODES] = {CUBLAS_DEFAULT_MATH, 
                           CUBLAS_PEDANTIC_MATH,
                           CUBLAS_TF32_TENSOR_OP_MATH,
                           CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION};

// CUBLAS COMPUTE TYPES (for cublasGemmEx and cublasLtMatmul)
#define NUM_CUBLAS_COMPUTE_TYPES 11
cublasComputeType_t cublasComputeTypes[NUM_CUBLAS_COMPUTE_TYPES] = {CUBLAS_COMPUTE_16F,
                                                CUBLAS_COMPUTE_16F_PEDANTIC,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_COMPUTE_32F_PEDANTIC,
                                                CUBLAS_COMPUTE_32F_FAST_16F,
                                                CUBLAS_COMPUTE_32F_FAST_16BF,
                                                CUBLAS_COMPUTE_32F_FAST_TF32,
                                                CUBLAS_COMPUTE_64F,
                                                CUBLAS_COMPUTE_64F_PEDANTIC,
                                                CUBLAS_COMPUTE_32I,
                                                CUBLAS_COMPUTE_32I_PEDANTIC};

cudaDataType dataTypes[3]       = {CUDA_R_16F, CUDA_R_32F, CUDA_R_64F}; 
const char* dataTypesStr[3]    = {"CUDA_R_16F", "CUDA_R_32F", "CUDA_R_64F"}; 
                                                 
// STRINGS FOR MESSAGES
// CUBLAS ALGORITHMS STRINGS
const char* cublasAlgsStr[NUM_CUBLAS_ALGS] = {"CUBLAS_GEMM_DEFAULT", 
                                              "CUBLAS_GEMM_DEFAULT_TENSOR_OP"};

// CUBLAS MATH MODES STRINGS
const char* cublasMathModesStr[NUM_CUBLAS_MATH_MODES] = {"CUBLAS_DEFAULT_MATH", 
                                   "CUBLAS_PEDANTIC_MATH",
                                   "CUBLAS_TF32_TENSOR_OP_MATH",
                                   "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION"};

// CUBLAS COMPUTE TYPES STRINGS (for cublasGemmEx and cublasLtMatmul)
const char* cublasComputeTypesStr[NUM_CUBLAS_COMPUTE_TYPES] = {"CUBLAS_COMPUTE_16F",
                                        "CUBLAS_COMPUTE_16F_PEDANTIC",
                                        "CUBLAS_COMPUTE_32F",
                                        "CUBLAS_COMPUTE_32F_PEDANTIC",
                                        "CUBLAS_COMPUTE_32F_FAST_16F",
                                        "CUBLAS_COMPUTE_32F_FAST_16BF",
                                        "CUBLAS_COMPUTE_32F_FAST_TF32",
                                        "CUBLAS_COMPUTE_64F",
                                        "CUBLAS_COMPUTE_64F_PEDANTIC",
                                        "CUBLAS_COMPUTE_32I",
                                        "CUBLAS_COMPUTE_32I_PEDANTIC"};

const char* cblasDataTypesStr[2] = {"float", "double"};

int log2i(int val){
    int r = 0;
    while (val >>= 1) ++r;
    return r;
}

int hmap(int b){
    return log2i(b) - 4;    
}

int cpuhmap(int b){
    return log2i(b) - 5;
}

void printDefines(const char** opts, int n, const char *msg){
    printf("%s:\n", msg);
    for(int i=0; i<n; ++i){
       cout << i << " = " << opts[i] << endl;  
    }
}

void printArgsInfo(){
    printDefines(cublasComputeTypesStr, NUM_CUBLAS_COMPUTE_TYPES, "CUBLAS Compute Types (comptype)");
}

template <typename T>
void print_matrix(T *mat, int M, int N, const char *msg){
    if(M <= LIM_PRINT_N && N <= LIM_PRINT_N){
        printf("%s:\n", msg);
        for(int i=0; i<M; ++i){
            for(int j=0; j<N; ++j){
                printf("%6.3f ", (float)mat[i*M + j]);
            }
            printf("\n");
        }
    }
}

static void cpuGemm(int n, CTYPE alpha, const ATYPE *A, const BTYPE *B, CTYPE beta, CTYPE *C){
    if(n > LIM_CHECK_N){ 
      return; 
    }
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float prod = 0;
            for (int k = 0; k < n; ++k) {
                prod += (float)A[k * n + i] * (float)B[j * n + k];
            }
            C[j * n + i] = (CTYPE) (float)((float)alpha * (float)prod + (float)beta * (float)C[j * n + i]);
        }
    }
}

template <typename T>
void fillMatrixRand(T *m, unsigned long nelem){
    #pragma omp parallel
    {
        random_device rd;
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<> dis(0.0, 1.0);
        long seg = (nelem+nt-1)/nt;
        long start = tid*seg;
        long end = start + seg;
        for(unsigned long i = start; i < nelem && i < end; i++){
            m[i] = (T)dis(gen);
        }
    }
}

template <typename T1, typename T2>
void copyMatrix(T1 *mTo, T2 *mFrom, int n){
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            long q = i*n+j;
            mTo[q] = (T1)mFrom[q];
        }
    }
}

template <typename T>
double computeMaxError(T *goldC, CTYPE *C, int N){
    double maxErr = 0.0;
    long nelem = (long)N*N;
    if(N <= LIM_CHECK_N && goldC){
        for(long i = 0; i < nelem; ++i){
            double g = (double)goldC[i];
            double c = (double)C[i];
            double denom = fabs(g);
            double err = (denom > 1e-12) ? fabs(g - c)/denom : fabs(g - c);
            if(err > maxErr){
                maxErr = err;
            }
        }
    }
    return maxErr;
}

template <typename T>
T* cblas_compute(int N, int reps, unsigned long nelem, CTYPE alpha, CTYPE beta, ATYPE *h_A, BTYPE *h_B, const char *dtypeCPU, bool verb){
    double TFLOP = 2.0*(double)N*(double)N*(double)N * 1E-12;
    double t1 = omp_get_wtime();
    T *cblasA = (T*)(malloc(nelem * sizeof(T)));
    T *cblasB = (T*)(malloc(nelem * sizeof(T)));
    T *cblasC = (T*)(malloc(nelem * sizeof(T)));
    double t2 = omp_get_wtime();
    (void)t1; (void)t2; // silence warnings if unused
    t1 = omp_get_wtime();
    copyMatrix<T, ATYPE>(cblasA, h_A, N);
    copyMatrix<T, BTYPE>(cblasB, h_B, N);
    t2 = omp_get_wtime();
    t1 = omp_get_wtime();
    if(verb){
        printf("[CBLAS] CPU GEMM (%3i reps, %6s)......", reps, dtypeCPU); fflush(stdout);
    }
    for(int q=0; q<reps; q++){
        #ifdef CPUFP64
          cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,alpha,cblasA,N,cblasB,N,beta,cblasC,N);
        #else
          cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,alpha,cblasA,N,cblasB,N,beta,cblasC,N);
        #endif
    }
    t2 = omp_get_wtime();
    double avgtime = (t2-t1)/(double)reps;
    double cpuTFLOPS = TFLOP/avgtime;
    if(verb){
        printf("done: %f secs [%f TFLOPS] (average of %i reps)\n\n", avgtime, cpuTFLOPS, reps); fflush(stdout);
    }
    print_matrix<T>(cblasC, N, N, "RESULT MAT C (CPU)");
    return cblasC;
}

