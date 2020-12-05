#ifndef _WRAPCUDA_H_
# define _WRAPCUDA_H_
#include <cuda.h>
#include <driver_types.h>

// In case you want to intercept, the callbacks need the same type/parameters as the real functions
typedef CUresult CUDAAPI (*fnMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult CUDAAPI (*fnMemFree)(CUdeviceptr dptr);
typedef CUresult CUDAAPI (*fnCtxGetCurrent)(CUcontext *pctx);
typedef CUresult CUDAAPI (*fnCtxSetCurrent)(CUcontext ctx);
typedef CUresult CUDAAPI (*fnCtxDestroy)(CUcontext ctx);
typedef CUresult CUDAAPI (*fnCuLaunchKernel)(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra);

/* Helper define for creating function pointer type and define the
 * function itself
 */
#define TYPEDEF_AND_CONFIG_FN(retval, fn_name, params)\
	typedef retval (*fn_ ## fn_name) params;\
	retval fn_name params;

/* Interposed function declarations */
extern "C"{
/* CUDA Driver Kernel API */
TYPEDEF_AND_CONFIG_FN(CUresult, cuLaunchKernel, (CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra));

/* CUDA Memory management Driver API functions */
TYPEDEF_AND_CONFIG_FN(CUresult, cuMemAlloc_v2, (CUdeviceptr* dptr, size_t bytesize));
TYPEDEF_AND_CONFIG_FN(CUresult, cuMemFree_v2, (CUdeviceptr dptr));
TYPEDEF_AND_CONFIG_FN(CUresult, cuMemGetInfo, ( size_t* free, size_t* total ));

// No need for v2
TYPEDEF_AND_CONFIG_FN(CUresult, cuMemHostAlloc, ( void** pp, size_t bytesize, unsigned int  Flags ));

/* Memcpy H<->D */
TYPEDEF_AND_CONFIG_FN(CUresult, cuMemcpyHtoDAsync_v2, ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream ));
/* Memset H<->D*/
// N 32-bit values to ui
TYPEDEF_AND_CONFIG_FN(CUresult, cuMemsetD32_v2, ( CUdeviceptr dstDevice, unsigned int  ui, size_t N ));

/* Memcpy TODO */
CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount );
CUresult cuMemcpyDtoH ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount );
CUresult cuMemcpyDtoHAsync ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream );
CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount );// Infers type, synchronous



/* Driver Device Management API */


} // extern "C"

#endif /* _WRAPCUDA_H_ */
