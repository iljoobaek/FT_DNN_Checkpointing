#include <assert.h>				// assert()
#include <sys/types.h> 			// off_t
#include <sys/mman.h>
#include <sys/stat.h>           // for mode constants
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>              // for O_ constants, such as "O_RDWR"
#include <string.h>				// memset(), strncpy
#include <stdio.h>				// *printf, perror
#include <stdint.h>				// int64_t
#include <semaphore.h>			// sem_t, sem_*()
#include "mid_structs.h"		// global_jobs_t, GLOBAL_JOBS_MAX*, JOB_MEM_NAME_MAX_LEN
#include "mid_common.h"

int init_global_jobs(int *fd, global_jobs_t **addr, bool init_flag)
{
	// Get fd for shared GLOBAL_MEM_SIZE memory region
    errno = 0;              //  automatically set when error occurs
    // open/create POSIX shared memory; if it doesn't exist, create one.
    *fd = shm_init(GLOBAL_MEM_NAME, GLOBAL_MEM_SIZE);
    if (*fd == -1){
        perror("[Error] in mmap_global_jobs_queue: shm_init failed");
        return -1;
    }

    *addr = (global_jobs_t *)mmap(  NULL,
                        GLOBAL_MEM_SIZE,
                        PROT_READ|PROT_WRITE,
                        MAP_SHARED,
                        *fd,
                        0);

    if (*addr == MAP_FAILED) {
        perror("[Error] in mmap_global_jobs_queue: mmap");
        return -1;
    }

	/* Initialize the global_jobs_t struct in shared memory */
	if (init_flag) {
		global_jobs_t *gj = *addr;
		gj->is_active = 1;
		gj->total_count = 0;
		// Zero out the job_shm_names array
		memset(gj->job_shm_names, 0, GLOBAL_JOBS_MAX_JOBS * JOB_MEM_NAME_MAX_LEN);

		// Initialize the global lock on the job_shm_names list
		pthread_mutexattr_t mutex_attr;

		// Initialize requests_q_lock to be PROCESS_SHARED
		(void) pthread_mutexattr_init(&mutex_attr);
		(void) pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);

		(void) pthread_mutex_init(&(gj->requests_q_lock), &mutex_attr);
	}

	return 0;
}

// helper functions for creating space in shared memory:
// shm_init: returns the file descriptor after creating a shared memory. returns
//  -1 on error;
int shm_init(const char *name, off_t init_size){
    errno = 0;              //  automatically set when error occurs
    // open/create POSIX shared memory; if it doesn't exist, create one.
    int fd;
    fd = shm_open(name, O_RDWR, 0660);
    if (errno == ENOENT){
        fd = shm_open(name, O_RDWR|O_CREAT, 0660);

		// Init shm to have size of job_t struct
		if (ftruncate(fd, init_size) != 0) {
			perror("ftruncate");
		}
    }
    if (fd == -1){
        perror("[Error] in shm_init: shm_open");
        return -1;
    }

    return fd;
}

/* shm_destroy: close the file descriptor, and unlink shared memory;
* returns 0 if successful, -1 otherwise
*/
int shm_destroy(const char *name, int fd) {
    // close the file descriptor
    if (close(fd)){
        perror("[Error] in shm_destroy: closing file descriptor");
        return -1;
    }
    // unlink shared memory
    if (shm_unlink(name)){
        perror("[Error] in shm_destroy: unlinking shm");
        return -1;
    }
    return 0;
}

/* Note: Not thread-safe, assumes only one job queued per tid */
int get_shared_job(const char *name, job_t **save_job)
{
	// Get fd for shared memory region designated by name
    errno = 0;              //  automatically set when error occurs
    int fd = shm_init(name, sizeof(job_t));
    if (fd == -1){
        perror("[Error] in get_shared_job: shm_init failed");
        return -1;
    }

	// Mmap the job_t object stored in shared memory
	job_t *shm_job = (job_t *)
						mmap(  NULL,
                        sizeof(job_t),
                        PROT_READ|PROT_WRITE,
                        MAP_SHARED,
                        fd,
                        0);

    if (shm_job == MAP_FAILED) {
        perror("[Error] in mmap_global_jobs_queue: mmap");
        return -1;
    }

	// Client opened the shared memory, it should still have a reference to
	// This only destroys local connection to shared memory object
	shm_destroy(name, fd);
	*save_job = shm_job;
 	return 0;
}

bool jobs_equal(job_t *a, job_t *b) {
	return ((a && b) \
			&& a->tid == b->tid \
			&& (strcmp(a->job_name, b->job_name) == 0));
}

int build_job(pid_t pid, pid_t tid, const char *job_name,
						int64_t slacktime_us, bool noslack_flag,
						bool shareable_flag,
						uint64_t required_mem_b,
						enum job_type req_type,
						job_t *shm_job) {
	// Fail on bad mapped job
	if (!shm_job) return -1;

	// Init job with metadata
	shm_job->pid = pid;
	shm_job->tid = tid;
	strncpy(shm_job->job_name, job_name, JOB_MEM_NAME_MAX_LEN);
	shm_job->slacktime_us = slacktime_us;
	shm_job->noslack_flag = noslack_flag;
	shm_job->shareable_flag = shareable_flag;
	shm_job->required_mem_b = required_mem_b;
	shm_job->req_type = req_type;

	// Init client-server semaphore and state
	int pshared = 1;
	if (sem_init(&(shm_job->client_wake), pshared, 0U)) {
		fprintf(stderr, "Failed to init semaphore for shared job!\n");
		return -1;
	}
	shm_job->client_exec_allowed = true;

 	return 0;
}

int build_shared_job(pid_t pid, pid_t tid, const char *job_name,
						int64_t slacktime_us, bool noslack_flag,
						bool shareable_flag,
						uint64_t required_mem_b,
						enum job_type req_type,
						job_t **save_new_job,
						char *copy_shm_name) {
	// First construct name for shared memory region
	char job_shm_name[JOB_MEM_NAME_MAX_LEN];
	sprintf(job_shm_name, "%s_%d", job_name, tid);

	// Get fd for shared memory region designated
    errno = 0;              //  automatically set when error occurs
    int fd = shm_init(job_shm_name, sizeof(job_t));
    if (fd == -1){
        perror("[Error] in build_shared_job: shm_init failed");
        return -1;
    }

	// Mmap the job_t object stored in shared memory
	job_t *shm_job = (job_t *)
						mmap(  NULL,
                        sizeof(job_t),
                        PROT_READ|PROT_WRITE,
                        MAP_SHARED,
                        fd,
                        0);

	// Close opened fd now that job is mapped
	DEBUG_FN(close, fd);

    if (shm_job == MAP_FAILED) {
        perror("[Error] in mmap_global_jobs_queue: mmap");
        return -1;
    }

	// Save job and job_shm_name for caller
	DEBUG_FN(build_job, pid, tid, job_name,
						slacktime_us, noslack_flag, shareable_flag,
						required_mem_b,
						req_type, shm_job);
	*save_new_job = shm_job;
	strncpy(copy_shm_name, job_shm_name, JOB_MEM_NAME_MAX_LEN);
	return 0;
}

/*
 * Release memory mapped for shared_job pointed to by shared_job_ptr
 * Return 0 on success, negative on error
 */
int destroy_shared_job(job_t **shared_job_ptr) {
	if (shared_job_ptr && *shared_job_ptr) {
		int res;
		if ((res = munmap(*shared_job_ptr, sizeof(job_t))) == 0) {
			*shared_job_ptr = NULL;
		}
		return res;
	}
	return -2;
}
