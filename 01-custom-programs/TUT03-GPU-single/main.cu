#include <cuda.h>
#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#define BSIZE2D 32

void explore_gpus();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// GPU matmul global mem
__global__ void kernel_matmul(int n, float *a, float *b, float *c){
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0.0f;
	for(int k=0; k<n; ++k){
		sum += a[ty*n + k]*b[k*n + tx];
	}
	c[ty*n + tx] = sum;
}

// GPU matmul shared mem 
__global__ void kernel_matmulsm(int n, float *a, float *b, float *c){
	__shared__ float as[BSIZE2D*BSIZE2D];	
	__shared__ float bs[BSIZE2D*BSIZE2D];	
	__shared__ float cs[BSIZE2D*BSIZE2D];	

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int ltx = threadIdx.x;
	int lty = threadIdx.y;

    cs[lty*BSIZE2D + ltx] = 0;
	// (1) hacer 'k' veces la version bloque 
	for(int k=0; k<n; k=k+BSIZE2D){
	// 	(a) cargar datos en as,bs. Escribir resultados en cs
	//	(b) sincorinizar la carga en as, bs, antes de calcular cs.
		as[lty*BSIZE2D + ltx] = a[ty*n + (k + ltx)];
		bs[lty*BSIZE2D + ltx] = b[(k + lty)*n + tx];
		__syncthreads();
		for(int r=0; r<BSIZE2D; ++r){
			cs[lty*BSIZE2D + ltx] += as[lty*BSIZE2D + r]*bs[r*BSIZE2D + ltx];
		}
		__syncthreads();
	}
	// (2) escribir cs en c
    c[ty*n + tx] = cs[lty*BSIZE2D + ltx];
}

void initmat(int n, float *m){
    #pragma omp parallel for
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            m[i*n + j] = i/100.0f;
        }
    }
}

void printmat(int n, float *m, const char* msg){
    printf("%s\n", msg);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            printf("%.2f ", m[i*n + j]);
        }
        printf("\n");
    }
}

int verify(int n, float *a, float *b, float *c, float *cgold){
    float error = 0.01f;
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            float sum = 0.0f;
            for(int k=0; k<n; ++k){
                sum += a[i*n + k]*b[k*n + j];
            }
            cgold[i*n + j] = sum;
            if(fabs(c[i*n + j] - cgold[i*n + j]) >= error){
                fprintf(stderr, "error: c[%i][%i] ---> c %f    cgold %f\n", i, j, c[i*n+j], cgold[i*n+j]);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char **argv){
    printf("GPU MATMUL\n");
    if(argc != 4){
        fprintf(stderr, "run as ./prog dev n mode\nmode:\n\t0 global mem\n\t1 shared mem\n");
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    int n = atoi(argv[2]);
    int mode = atoi(argv[3]);
    printf("size: %i x %i\nmode %i\n", n, n, mode);
    float msecs = 0.0f;

    printf("Exploring GPUs"); fflush(stdout);
    explore_gpus();

    printf("Choosing GPU %i\n", dev);
    gpuErrchk(cudaSetDevice(dev));

    // (1) creando matrices en host
    float *a = new float[n*n];
    float *b = new float[n*n];
    float *c = new float[n*n];
    float *cgold = new float[n*n];
    printf("initializing A and B......."); fflush(stdout);
    initmat(n, a);
    initmat(n, b);
    if(n < 64){
        printmat(n, a, "mat a");
        printmat(n, b, "mat b");
    }
    printf("done\n"); fflush(stdout);

    // (2) copiar matrices en device
    float *ad, *bd, *cd;
    gpuErrchk(cudaMalloc(&ad, sizeof(float)*n*n));
    gpuErrchk(cudaMalloc(&bd, sizeof(float)*n*n));
    gpuErrchk(cudaMalloc(&cd, sizeof(float)*n*n));
    gpuErrchk(cudaMemcpy(ad, a, sizeof(float)*n*n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bd, b, sizeof(float)*n*n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cd, c, sizeof(float)*n*n, cudaMemcpyHostToDevice));

    // (3) run matmul en GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(BSIZE2D, BSIZE2D, 1);
    dim3 grid((n+BSIZE2D-1)/BSIZE2D, (n+BSIZE2D-1)/BSIZE2D, 1); 
    cudaEventRecord(start);
    if(mode == 0){
        printf("matmul global mem.........."); fflush(stdout);
        kernel_matmul<<<grid, block>>>(n, ad, bd, cd);
    }
    else if(mode == 1){
        printf("matmul shared mem.........."); fflush(stdout);
        kernel_matmulsm<<<grid, block>>>(n, ad, bd, cd);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecs, start, stop);
    printf("done: time: %f secs\n", msecs/1000.0f);

    // (4) copiar resultado a host
    printf("copying result to host....."); fflush(stdout);
    gpuErrchk(cudaMemcpy(c, cd, sizeof(float)*n*n, cudaMemcpyDeviceToHost));
    printf("done\n"); fflush(stdout);

    if(n < 50){
        printmat(n, c, "mat c");
    }

    // (5) verificar resultado con calculo en CPU
    printf("verifying result..........."); fflush(stdout);
    if(n <= 512){
        if(!verify(n, a, b, c, cgold)){
            printf("failed\n"); fflush(stdout);
        }
        else{
            printf("pass\n"); fflush(stdout);
        }
    }
    printf("done\n"); fflush(stdout);
    exit(EXIT_SUCCESS);
}


/* check nvml result */
int nvml_check(nvmlReturn_t r, const char* mesg){
	if(r != NVML_SUCCESS){
		if(r == NVML_ERROR_UNINITIALIZED)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_UNINITIALIZED\n", mesg);
		else if(r == NVML_ERROR_INVALID_ARGUMENT)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_INVALID_ARGUMENT\n", mesg);
		else if(r == NVML_ERROR_NOT_SUPPORTED)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_NOT_SUPPORTED\n", mesg);
		else if(r == NVML_ERROR_GPU_IS_LOST)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_GPU_IS_LOST\n", mesg);
		else if(r == NVML_ERROR_UNKNOWN)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_UNKNOWN\n", mesg);
		else
			fprintf(stderr, "nvml error: %s: code not listed\n", mesg);
		return 0;
	}
	return 1;
}

/* pick the idlest 'n' gpus */
void explore_gpus(){ 
	/* structs for handling GPU queries error codes */
	nvmlReturn_t r;
	/* some function variables */
	unsigned int devcount, i, u;
	/* struct with GPU information */
	char version[80];
	/* init nvml library for GPU queries */
	r = nvmlInit(); 
	nvml_check(r, "nvmlInit");

	/* nvml: get driver version */
	r = nvmlSystemGetDriverVersion(version, 80); 
	nvml_check(r, "nvmlSystemGetDriverVersion");
	printf("\n\tDriver version: %s \n", version);

	/* get number of devices */
	r = nvmlDeviceGetCount(&devcount); 
	nvml_check(r, "nvmlDeviceGetCount");
	printf("\tNUM GPUS = %d\n", devcount);

	/* get the information of each GPU */
	printf("\tListing devices:\n");
	for(i = 0; i < devcount; i++){
        unsigned int index;
		nvmlDevice_t dev;
		char name[64];
        char uuid[128];
		//nvmlComputeMode_t compute_mode;
		nvmlUtilization_t util;
		r = nvmlDeviceGetHandleByIndex(i, &dev); 
		nvml_check(r, "nvmlDeviceGetHandleByIndex");
		r = nvmlDeviceGetName(dev, name, sizeof(name)/sizeof(name[0])); 
		nvml_check(r, "nvmlDeviceGetName");
        r = nvmlDeviceGetIndex(dev, &index);
        r = nvmlDeviceGetUUID(dev, uuid, 128);
		printf("\t\tGPU%d %s, index=%i, UUID=%s", i, name, index, uuid);
		r = nvmlDeviceGetUtilizationRates(dev, &util); 
		u = nvml_check(r, "nvmlDeviceGetUtilizationRates");
		if(u){
			printf("  -> util = %i%%\n", util.gpu);
		}
	}
	r = nvmlShutdown();
	nvml_check(r, "nvmlShutdown");
	/* free the auxiliary gpu_t array */
}
