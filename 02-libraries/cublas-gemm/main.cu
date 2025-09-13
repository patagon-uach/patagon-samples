/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>       // <-- added
#include <cblas.h>
#include <omp.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define LIM_CHECK_N 4096
#define LIM_PRINT_N 32
// fraction error   1.0 is 100% 
#define TOLERR 0.0001
#ifdef CPUFP64
    typedef double CPUTYPE;
#else
    typedef float CPUTYPE;
#endif
#include "tools.h"

using namespace std;

int main(int argc, char **argv) {
  cublasStatus_t status;
  if(argc != 6){
      fprintf(stderr, "run as ./prog dev nt n comptype mode\n\n"
              "dev:      Device ID\n"
              "nt:       Number of CPU threads (accelerates data init and CPU mode)\n"
              "n:        Matrix size of n x n\n"
              "comptype: GPU CUBLAS mode\n"
              "mode:     CPU=0,  GPU=1\n\n");

      printArgsInfo();
      return EXIT_FAILURE;
  }
  float gputime_ms;
  int dev = atoi(argv[1]);
  int nt = atoi(argv[2]);
  int N = atoi(argv[3]);
  int comptype = atoi(argv[4]);
  int mode = atoi(argv[5]);
  printf("\n*********************************************\n"
         "******** CUBLAS Example by Temporal *********\n"
         "*********************************************\n\n");
  printf("dev=%i, nt=%i, n=%i, cublasType=%i, <mode = %i -> %s>\n\n", dev, nt, N, comptype, mode, mode == 0? "CPU" : "GPU");
  // host pointers
  ATYPE *h_A;
  BTYPE *h_B;
  CTYPE *h_C;
  CPUTYPE *cblasC = nullptr;
  // device pointers
  ATYPE *d_A = 0;
  BTYPE *d_B = 0;
  CTYPE *d_C = 0;
  // constants
  CTYPE alpha = 1.0f;
  CTYPE beta = 0.0f;
  // number of elements
  unsigned long nelem = (unsigned long)N * (unsigned long)N;
  double GBytesUsed = (double)nelem*(sizeof(ATYPE) + sizeof(BTYPE) + sizeof(CTYPE))/1e9;
  double t1, t2;
  double TFLOP = 2.0*(double)N*(double)N*(double)N * 1E-12;
  int bitsA = sizeof(ATYPE)*8;
  int bitsB = sizeof(BTYPE)*8;
  int bitsC = sizeof(CTYPE)*8;
  int bitsCPU = sizeof(CPUTYPE)*8;

  cudaDataType dtypeA = dataTypes[hmap(bitsA)];
  cudaDataType dtypeB = dataTypes[hmap(bitsB)];
  cudaDataType dtypeC = dataTypes[hmap(bitsC)];
  const char* dtypeAStr = dataTypesStr[hmap(bitsA)];
  const char* dtypeBStr = dataTypesStr[hmap(bitsB)];
  const char* dtypeCStr = dataTypesStr[hmap(bitsC)];
  const char* dtypeCPU = cblasDataTypesStr[cpuhmap(bitsCPU)];

  gpuErrchk(cudaSetDevice(dev));
  print_gpu_specs(dev);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasHandle_t handle;
  omp_set_num_threads(nt);
  printf("Matrix size %i x %i --> %lu elements\n"
          "GPU: A FP%i (%10s), B FP%i (%10s), C FP%i (%10s)\n"
          "CPU: A FP%i (%10s), B FP%i (%10s), C FP%i (%10s)\n\n", 
          N, N, nelem,  
          bitsA, dtypeAStr,
          bitsB, dtypeBStr,
          bitsC, dtypeCStr, 
          bitsCPU, dtypeCPU,
          bitsCPU, dtypeCPU,
          bitsCPU, dtypeCPU);

  printf("GPU Mem used...................%f GB\n", GBytesUsed); fflush(stdout);
  printf("Pinned Mem.....................");
  #ifdef PINNED
    printf("True\n");
  #else
    printf("False\n");
  #endif

  /* 1) Initialize CUBLAS */
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  /* 2) Set math mode */
  printf("Compute Type...................%s\n\n", cublasComputeTypesStr[comptype]);
  // (optional) set math mode here if desired with cublasSetMathMode

  /* 3) Allocate and fill host memory for the matrices */
  printf("Host mallocs A B C............."); fflush(stdout);
  t1 = omp_get_wtime();
  #ifdef PINNED
      gpuErrchk(cudaMallocHost((void**)&h_A, nelem*sizeof(h_A[0])));
      gpuErrchk(cudaMallocHost((void**)&h_B, nelem*sizeof(h_B[0])));
      gpuErrchk(cudaMallocHost((void**)&h_C, nelem*sizeof(h_C[0])));
  #else
      h_A = (ATYPE*)(malloc(nelem * sizeof(h_A[0])));
      h_B = (BTYPE*)(malloc(nelem * sizeof(h_B[0])));
      h_C = (CTYPE*)(malloc(nelem * sizeof(h_C[0])));
      if(!h_A || !h_B || !h_C){ fprintf(stderr, "Host malloc failed\n"); return EXIT_FAILURE; }
  #endif

  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);
  printf("Filling matrices in Host......."); fflush(stdout);
  t1 = omp_get_wtime();
  fillMatrixRand<ATYPE>(h_A, nelem);
  fillMatrixRand<BTYPE>(h_B, nelem);
  fillMatrixRand<CTYPE>(h_C, nelem);
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);
  print_matrix<ATYPE>(h_A, N, N, "MAT A");
  print_matrix<BTYPE>(h_B, N, N, "MAT B");

  /* 4) Allocate device memory for the matrices */
  if(mode==1){
      printf("Device mallocs A B C..........."); fflush(stdout);
      t1 = omp_get_wtime();
      if (cudaMalloc(reinterpret_cast<void **>(&d_A), nelem * sizeof(d_A[0])) != cudaSuccess) {
            fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
            return EXIT_FAILURE;
      }

      if (cudaMalloc(reinterpret_cast<void **>(&d_B), nelem * sizeof(d_B[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
      }

      if (cudaMalloc(reinterpret_cast<void **>(&d_C), nelem * sizeof(d_C[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
      }
      t2 = omp_get_wtime();
      printf("done: %f secs\n", t2-t1); fflush(stdout);

      /* 5) Initialize the device matrices with the host matrices */
      // Use cudaMemcpy with size_t byte counts (cublasSetVector uses 32-bit n and overflows here)
      printf("Host -> Device memcpy A........"); fflush(stdout);
      t1 = omp_get_wtime();
      {
        size_t bytesA = (size_t)nelem * sizeof(h_A[0]);
        gpuErrchk(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
      }
      gpuErrchk(cudaDeviceSynchronize());
      t2 = omp_get_wtime();
      printf("done: %f secs (%f GB/sec)\n", t2-t1, (nelem*sizeof(h_A[0]))/(1e9 * (t2-t1))); fflush(stdout);

      printf("Host -> Device memcpy B........"); fflush(stdout);
      t1 = omp_get_wtime();
      {
        size_t bytesB = (size_t)nelem * sizeof(h_B[0]);
        gpuErrchk(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
      }
      gpuErrchk(cudaDeviceSynchronize());
      t2 = omp_get_wtime();
      printf("done: %f secs (%f GB/sec)\n", t2-t1, (nelem*sizeof(h_B[0]))/(1e9 * (t2-t1))); fflush(stdout);

      printf("Host -> Device memcpy C........"); fflush(stdout);
      t1 = omp_get_wtime();
      {
        size_t bytesC = (size_t)nelem * sizeof(h_C[0]);
        gpuErrchk(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));
      }
      gpuErrchk(cudaDeviceSynchronize());
      t2 = omp_get_wtime();
      printf("done: %f secs (%f GB/sec)\n\n", t2-t1, (nelem*sizeof(h_C[0]))/(1e9 * (t2-t1))); fflush(stdout);
  }

  /* 6) GEMM -> GPU CUBLAS */
  if(mode==1){
      printf("[CUBLAS] GPU GEMM.............."); fflush(stdout);
      gpuErrchk(cudaEventRecord(start));
      status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                        d_A, dtypeA, N,
                                        d_B, dtypeB, N,
                              &beta,    d_C, dtypeC, N, cublasComputeTypes[comptype],  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      if(status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
      }
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(stop));
      gpuErrchk(cudaEventSynchronize(stop));
      gpuErrchk(cudaEventElapsedTime(&gputime_ms, start, stop));
      double gpuTFLOPS = TFLOP/(gputime_ms/1000.0);
      printf("done: %f secs [%f TFLOPS]\n", gputime_ms/1000.0, gpuTFLOPS); fflush(stdout);
  }

  /* 7) GEMM -> CPU BASIC */
  if(mode == 0){
      cblasC = cblas_compute<CPUTYPE>(N, nelem, alpha, beta, h_A, h_B, dtypeCPU, true); 
  }

  /* 8) Read the result back */
  if(mode == 1){
      printf("Device -> Host memcpy C........"); fflush(stdout);
      t1 = omp_get_wtime();
      {
        size_t bytesC = (size_t)nelem * sizeof(h_C[0]);
        gpuErrchk(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
      }
      t2 = omp_get_wtime();
      printf("done: %f secs (%f GB/sec)\n", t2-t1, (nelem*sizeof(h_C[0]))/(1e9*(t2-t1))); fflush(stdout);
      print_matrix<CTYPE>(h_C, N, N, "RESULT MAT C (GPU)");
  }

  /* 9) Check result against reference */
  if(mode == 1){
      printf("Verify result.................."); fflush(stdout);
      t1 = omp_get_wtime();
      if(N < LIM_CHECK_N){
          cblasC = cblas_compute<CPUTYPE>(N, nelem, alpha, beta, h_A, h_B, dtypeCPU, false); 
      }
      double maxError = computeMaxError<CPUTYPE>(cblasC, h_C, N); 
      t2 = omp_get_wtime();
      printf("done: %f secs (maxError = %f%%, TOL = %f%%)\n%s\n\n", t2-t1,
              maxError*100.0, TOLERR*100.0, 
              maxError <= TOLERR ? (const char*)"pass" : (const char*) "failed"); fflush(stdout);
  }

  /* 10) Memory clean up */
  #ifdef PINNED
      cudaFreeHost(h_A);
      cudaFreeHost(h_B);
      cudaFreeHost(h_C);
  #else
      free(h_A);
      free(h_B);
      free(h_C);
  #endif

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* 11) Shutdown */
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }
}

