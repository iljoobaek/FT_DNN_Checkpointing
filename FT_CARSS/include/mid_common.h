#ifndef COMMON_H
#define COMMON_H

#include <stdio.h> 		// *printf
#include <stdint.h>				// int64_t
#include <assert.h>		// assert()
#include "mid_structs.h" // global_jobs_t
// ------------------------ Defines ------------------------
#define DEBUG_FN(fn, ...) \
{\
	int res;\
	if ((res = fn(__VA_ARGS__)) < 0) { \
		fprintf(stderr, "DEBUG_FN: %s returned %d!\n", #fn, res);\
		assert(false);\
	}\
}


#define GLOBAL_MEM_NAME "global_mem"
#define JOB_MEM_NAME "job_mem"
#define DATA_MEM_NAME "data_mem"

#define SLEEP_MICROSECONDS (unsigned int)50
#define GLOBAL_MEM_SIZE sizeof(global_jobs_t)

/* Jobs will be saved in shared memory objects named like: <name>_<tid> */
#define JOB_SHM_NAME(tid, name_s) #name_s ## _ ## #tid

#ifdef __cplusplus
extern "C" {
#endif
int shm_init(const char *name, off_t init_size);
int shm_destroy(const char *name, int fd);
int init_global_jobs(int *fd, global_jobs_t **addr, bool init_flag);
int get_shared_job(const char *name, job_t **save_job);

bool jobs_equal(job_t *a, job_t *b);
int build_shared_job(pid_t pid, pid_t tid, const char *job_name,
						int64_t slacktime_us, bool noslack_flag,
						bool shareable_flag,
						uint64_t required_mem_b,
						enum job_type req_type,
						job_t **save_new_job,
						char *copy_shm_name);
int destroy_shared_job(job_t **shared_job_ptr);
#ifdef __cplusplus
}
#endif

#endif
