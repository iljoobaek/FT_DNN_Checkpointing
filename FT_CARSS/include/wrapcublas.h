#ifndef _WRAPCUBLAS_H_
# define _WRAPCUBLAS_H_
#define CUBLASAPI
#include <cublas_api.h>

typedef cublasStatus_t (*fn_cublasSgemm_v2)(cublasHandle_t, 
                                                      cublasOperation_t,
                                                      cublasOperation_t, 
                                                      int,
                                                      int,
                                                      int,
                                                      const float *,
                                                      const float *, 
                                                      int,
                                                      const float *,
                                                      int, 
                                                      const float *, 
                                                      float *,
                                                      int);

typedef cublasStatus_t (*fn_cublasDgemm_v2)(cublasHandle_t, 
														  cublasOperation_t,
														  cublasOperation_t, 
														  int,
														  int,
														  int,
														  const double *, 
														  const double *, 
														  int,
														  const double *,
														  int, 
														  const double *, 
														  double *,
														  int);

/* Interposed cublas functions */
extern "C"
{
/* Note: we actually interpose the "v2" functions of cublas API because
 * these replace the v1 versions at link-time.
 */
cublasStatus_t cublasCreate /*_v2*/ (cublasHandle_t *handle);
cublasStatus_t cublasDestroy /*_v2*/ (cublasHandle_t handle);

cublasStatus_t cublasSgemm_v2 (cublasHandle_t handle, 
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
                                                      int ldc);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, 
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
														  int ldc);
} /* extern "C" */
#endif /* _WRAPCUBLAS_H_ */

