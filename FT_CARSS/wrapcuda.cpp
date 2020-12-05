// This sample demonstrates a simple library to interpose CUDA symbols

#define __USE_GNU
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>    // getpid()
#include <signal.h>    // raise(), SIGSTP
#include <stdlib.h>    // exit()
#include <sys/types.h> // pid_t

#include <vector_types.h> // dim3
#include <driver_types.h> // cudaError_t
#include "wrapcuda.h"
#include "wrapcublas.h"   //cublas interposed fn's
#include "wraputil.h"     // queue_job_to_server
#include "mid_structs.h"  // job_t
#include "mid_common.h"  // JOB_MEM_NAME
#include "mid_queue.h" // *queue_*()

/* We want to interpose using LD_PRELOAD version 2 of cuBLAS API, so 
 * we DON'T include cublas.h
 */
// Note: v2 function declarations are also in cublas_api.h, but they won't bind 
// those in this file
#include <cublas_api.h>

// Keep copy of global_jobs_q
static job_t *global_jobs_q = NULL;
static int jobs_q_fd = -1;

// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface function
// For interposing dlopen(). Sell elf/dl-lib.c for the internal dlopen_mode interface function
extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

// We need to give the pre-processor a chance to replace a function, such as:
// cuMemAlloc => cuMemAlloc_v2
#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

// Hack for getting kernel name from CUfunction func
#define HACK_GET_KERN_NAME(func) *(const char **)((uintptr_t)func + 8)

// We need to interpose dlsym since anyone using dlopen+dlsym to get the CUDA driver symbols will bypass
// the hooking mechanism (this includes the CUDA runtime). Its tricky though, since if we replace the
// real dlsym with ours, we can't dlsym() the real dlsym. To get around that, call the 'private'
// libc interface called __libc_dlsym to get the real dlsym.
typedef void* (*fnDlsym)(void*, const char*);

static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

/* Interposed functions' real function handles */
static fnCuLaunchKernel realCLK_dr;
static fn_cublasSgemm_v2 real_cublasSgemm_v2;
static fn_cublasDgemm_v2 real_cublasDgemm_v2;

static fn_cuMemAlloc_v2 real_cuMemAlloc_v2;
static fn_cuMemFree_v2 real_cuMemFree_v2;
static fn_cuMemGetInfo real_cuMemGetInfo;
static fn_cuMemHostAlloc real_cuMemHostAlloc;
static fn_cuMemcpyHtoDAsync_v2 real_cuMemcpyHtoDAsync_v2;
static fn_cuMemsetD32_v2 real_cuMemsetD32_v2;
/* 
 * Helper functions 
 */

void hijack_handle(void *handle, const char *symbol, void **fn_ptr)
{
	// Presumably, a Python application is trying to locatesymbol 
	// so hijack the handle to retrieve would-be real fn, if possible
	dlerror();
	const char *error;
	void *real_fn = real_dlsym(handle, symbol);
	// Print error
	if ((error = dlerror()) != NULL) 
	{
		fprintf(stderr, "Hijacking would-be libcuda.so handle \
to retrieve real symbol (%s) failed! err: %s\n", symbol, error);
		// Should never reach here!
		exit(EXIT_FAILURE);
	}

	// Save real_fn into fn_ptr
	if (real_fn)
	{
		*fn_ptr = real_fn;
	}
}

/*
 ** Interposed Functions
 */
void* dlsym(void *handle, const char *symbol)
{
    // Early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }
	
	if (strcmp(symbol, CUDA_SYMBOL_STRING(cuLaunchKernel)) == 0) {
		if (handle != RTLD_DEFAULT && !realCLK_dr)
		{
			// Hijack the explicit handle opened to retrieve real function defn
			hijack_handle(handle, "cuLaunchKernel", (void**)&realCLK_dr);
		}
		return (void*)(&cuLaunchKernel);
	}
	else if (strcmp(symbol, CUDA_SYMBOL_STRING(cublasSgemm_v2)) == 0) {
		if (handle != RTLD_DEFAULT && !real_cublasSgemm_v2)
		{
			hijack_handle(handle, CUDA_SYMBOL_STRING(cublasSgemm_v2), (void**)&real_cublasSgemm_v2);
		}
	    return (void*)(&cublasSgemm_v2);
	}
	else if (strcmp(symbol, CUDA_SYMBOL_STRING(cublasDgemm_v2)) == 0) {
		if (handle != RTLD_DEFAULT && !real_cublasDgemm_v2)
		{
			hijack_handle(handle, CUDA_SYMBOL_STRING(cublasDgemm_v2), (void**)&real_cublasDgemm_v2);
		}
	    return (void*)(&cublasDgemm_v2);
	}
	//else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc_v2)) == 0) {
	//	if (handle != RTLD_DEFAULT && !real_cuMemAlloc_v2)
	//	{
	//		hijack_handle(handle, CUDA_SYMBOL_STRING(cuMemAlloc_v2), (void**)&real_cuMemAlloc_v2);
	//	}
	//    return (void*)(&cuMemAlloc_v2);
	//}
	//else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree_v2)) == 0) {
	//	if (handle != RTLD_DEFAULT && !real_cuMemFree_v2)
	//	{
	//		hijack_handle(handle, CUDA_SYMBOL_STRING(cuMemFree_v2), (void**)&real_cuMemFree_v2);
	//	}
	//    return (void*)(&cuMemFree_v2);
	//}
	else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemGetInfo)) == 0) {
		if (handle != RTLD_DEFAULT && !real_cuMemGetInfo)
		{
			hijack_handle(handle, CUDA_SYMBOL_STRING(cuMemGetInfo), (void**)&real_cuMemGetInfo);
		}
	    return (void*)(&cuMemGetInfo);
	}
	//else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemHostAlloc)) == 0) {
	//	if (handle != RTLD_DEFAULT && !real_cuMemHostAlloc)
	//	{
	//		hijack_handle(handle, CUDA_SYMBOL_STRING(cuMemHostAlloc), (void**)&real_cuMemHostAlloc);
	//	}
	//    return (void*)(&cuMemHostAlloc);
	//}
	//else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyHtoDAsync_v2)) == 0) {
	//	if (handle != RTLD_DEFAULT && !real_cuMemcpyHtoDAsync_v2)
	//	{
	//		hijack_handle(handle, CUDA_SYMBOL_STRING(cuMemcpyHtoDAsync_v2), (void**)&real_cuMemcpyHtoDAsync_v2);
	//	}
	//    return (void*)(&cuMemcpyHtoDAsync_v2);
	//}
	//else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemsetD32_v2)) == 0) {
	//	if (handle != RTLD_DEFAULT && !real_cuMemsetD32_v2)
	//	{
	//		hijack_handle(handle, CUDA_SYMBOL_STRING(cuMemsetD32_v2), (void**)&real_cuMemsetD32_v2);
	//	}
	//    return (void*)(&cuMemsetD32_v2);
	//}
	
	////else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxGetCurrent)) == 0) {
	////    return (void*)(&cuCtxGetCurrent);
	////}
	//else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxSetCurrent)) == 0) {
	//    return (void*)(&cuCtxSetCurrent);
	//}
	//else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxDestroy)) == 0) {
	//    return (void*)(&cuCtxDestroy);
	//}
    return (real_dlsym(handle, symbol));
}

/* Interposed CUDA Driver API functions */
extern "C" 
{

	CUresult CUDAAPI cuLaunchKernel(CUfunction f,
									unsigned int gridDimX,
									unsigned int gridDimY,
									unsigned int gridDimZ,
									unsigned int blockDimX,
									unsigned int blockDimY,
									unsigned int blockDimZ,
									unsigned int sharedMemBytes,
									CUstream hStream,
									void **kernelParams,
									void **extra)
	{
		fprintf(stderr, "Caught my cuLaunchKernel (Driver API)!!\n");

		if (!realCLK_dr)
		{
			// If caught dlsym call did not find realCLK_dr from caught 
			// manual-binding from within Python (presumably), then only chance 
			// left is to look for the real cuLaunchKernel symbol ourselves at 
			// runtime (only case should from a C++ application)
			const char *error;

			// Clear dlerror 
			dlerror();

			// Find CUDA Driver/C API's cuLaunchKernel definition
			// using real_dlsym. 
			realCLK_dr = (fnCuLaunchKernel)real_dlsym(RTLD_NEXT,
												 CUDA_SYMBOL_STRING("cuLaunchKernel"));

			fprintf(stderr, "realCLK_dr set to %p\n", (void*)realCLK_dr);
			// Print error
			if ((error = dlerror()) != NULL) 
			{
				fprintf(stderr, "%s\n", error);
				exit(EXIT_FAILURE);
			}
		}

		/*
		 * Package kernel, launch config, and kernel args for server
		 */
		// First, get the kernel symbol name using hack from
		// https://devtalk.nvidia.com/default/topic/821920/how-to-get-a-kernel-functions-name-through-its-pointer-/
		const char *kern_name = HACK_GET_KERN_NAME(f);	
		fprintf(stderr, "Retrieved kernel name from CUfunction f, %s\n", kern_name);

		return realCLK_dr(f, 
							gridDimX, gridDimY, gridDimZ,
							blockDimX, blockDimY, blockDimZ,
							sharedMemBytes, hStream, kernelParams,
							extra);
	}

	/* CUDA Memory management Driver API functions */
#define GENERATE_CUMEM_INTERCEPT(funcname, params, ...)   \
		CUresult funcname params                                    \
		{                                                                   \
			fprintf(stderr, "Caught %s (cuMem* Driver API)\n", #funcname); \
			/* First, check to see if we can find real defn if not already
			 * hijacked
			 */\
			if (!real_ ## funcname)\
			{\
				real_ ## funcname = (fn_ ## funcname)\
					real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname));\
				if (!real_ ## funcname)	\
				{\
					fprintf(stderr, "Failed to get real fn for %s\n", #funcname);	\
					exit(EXIT_FAILURE);\
				}\
			}\
			/* Next, ensure global memory mapping is init for global jobs queue */\
			if (!global_jobs_q) \
			{ \
				int res;\
				if ((res = mmap_global_jobs_queue(&jobs_q_fd, (void**)&global_jobs_q)) < 0)\
				{\
					fprintf(stderr, "Failed to mmap global jobs queue!\n");\
					exit(EXIT_FAILURE);\
				}\
			}\
			/* Queue job to server */\
			int res;\
			if ((res = queue_job_to_server(getpid(), #funcname, global_jobs_q)) < 0) \
			{\
				fprintf(stderr, "Failed to queue job to server!\n");\
				exit(EXIT_FAILURE);\
			}\
			/* Pause this thread, wait to be continued by middleware */ \
			fprintf(stderr, "Tensorflow thread raising SIGSTOP\n");\
			raise(SIGSTOP);\
			fprintf(stderr, "Tensorflow thread woke up! Continuing ... \n");\
			\
			/* Lastly, for now we can call the function from the client */\
			return real_ ## funcname(__VA_ARGS__);\
		}
	//GENERATE_CUMEM_INTERCEPT(cuMemAlloc_v2, (CUdeviceptr* dptr, size_t bytesize),\
	//										dptr, bytesize);
	//GENERATE_CUMEM_INTERCEPT(cuMemFree_v2, (CUdeviceptr dptr), dptr);
	//GENERATE_CUMEM_INTERCEPT(cuMemHostAlloc, ( void** pp, size_t bytesize, unsigned int  Flags ),\
	//										pp, bytesize, Flags);
	//GENERATE_CUMEM_INTERCEPT(cuMemGetInfo, ( size_t* free, size_t* total ),\
	//										free, total);
	//GENERATE_CUMEM_INTERCEPT(cuMemcpyHtoDAsync_v2, ( CUdeviceptr dstDevice, 
	//												 const void* srcHost, 
	//												 size_t ByteCount, 
	//												 CUstream hStream ),
	//											dstDevice, srcHost, ByteCount,
	//											hStream);	
	//GENERATE_CUMEM_INTERCEPT(cuMemsetD32_v2, ( CUdeviceptr dstDevice, unsigned int  ui, size_t N ),\
	//										dstDevice, ui, N);

	/* cuBLAS API interposition 
 	 * Intercepting these functions allows
	 * our middleware to get to run these functions within its own address-space.
	 */
	/* TODO: How handle cublas handle creation/destruction */
	cublasStatus_t cublasCreate_v2 (cublasHandle_t *handle)
	{
		return CUBLAS_STATUS_EXECUTION_FAILED;	
	}
	cublasStatus_t cublasDestroy_v2 (cublasHandle_t handle)
	{
		return CUBLAS_STATUS_EXECUTION_FAILED;	
	}

/* cuBLAS interposed functions */
#define GENERATE_CUBLAS_INTERCEPT(funcname, params, ...)   \
		cublasStatus_t funcname params                                    \
		{                                                                   \
			fprintf(stderr, "Caught %s (CUBLAS API)\n", #funcname); \
			/* First, check to see if we can find real defn if not already
			 * hijacked
			 */\
			if (!real_ ## funcname)\
			{\
				real_ ## funcname = (fn_ ## funcname)\
					real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname));\
			}\
			/* Lastly, for now we can call the function from the client */\
			return real_ ## funcname(__VA_ARGS__);\
		}

	// Single-precision gemm
	GENERATE_CUBLAS_INTERCEPT(cublasSgemm_v2, (cublasHandle_t handle, 
														  cublasOperation_t transa,
														  cublasOperation_t transb, 
														  int m,
														  int n,
														  int k,
														  const float *alpha, /* host or device pointer */  
														  const float *A, 
														  int lda,
														  const float *B,
														  int ldb, 
														  const float *beta, /* host or device pointer */  
														  float *C,
														  int ldc), 
										 handle, transa, transb, m, n, k, alpha,
										 A, lda, B, ldb, beta, C, ldc);

	// Double-precision gemm
	GENERATE_CUBLAS_INTERCEPT(cublasDgemm_v2, (cublasHandle_t handle, 
														  cublasOperation_t transa,
														  cublasOperation_t transb, 
														  int m,
														  int n,
														  int k,
														  const double *alpha, /* host or device pointer */  
														  const double *A, 
														  int lda,
														  const double *B,
														  int ldb, 
														  const double *beta, /* host or device pointer */  
														  double *C,
														  int ldc), 
										 handle, transa, transb, m, n, k, alpha,
										 A, lda, B, ldb, beta, C, ldc);
	// TODO: Any more gemm functions?
} /* extern "C" */

