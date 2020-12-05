#ifndef STRUCTS_H
#define STRUCTS_H

#include <pthread.h> // pthread_mutex_t
#include <semaphore.h> // sem_t
#include <time.h> // time_t
#include <stdint.h> // uint64_t
#include <string.h> //strcmp
#include <stdbool.h> // bool

#define MAX_NAME_LENGTH 80 

#define GLOBAL_JOBS_MAX_JOBS 100
#define JOB_MEM_NAME_MAX_LEN 100

// ------------------------ Data structures ------------------------
enum job_type {QUEUED, COMPLETED};

// struct for a job, or a single GPU request;
typedef struct job {
	// job metadata
	pid_t pid;						// process id of job
    pid_t tid;                      // tid (since client may be multithreaded)

	char job_name[MAX_NAME_LENGTH];
	int64_t slacktime_us;				// Est. time remaining before job's deadline after execution (us)
	bool noslack_flag;				// slacktime ignored if true
	bool shareable_flag;			// Can job execute on GPU with other shareable jobs from other pids?
	uint64_t required_mem_b;			// Worst-case job GPU memory requirement (bytes)
	enum job_type req_type;			// 'queued' or 'completed'
	
	// Client-side/server-side attrs - communication properties
	sem_t client_wake;				// Semaphore controlling when client can continue within a tag
	bool client_exec_allowed;		// flag determining whether client should execute when woken
} job_t;


// GLOBAL struct that keeps track of all the jobs submitted to middleware, and
//  the list of completed jobs with their results that are passed back to 
//  clients 

typedef struct global_jobs {
    int is_active;                  // 1 after the server is started;
    int total_count;

	// Statically alloc'd array of client-created shared memory
	// object names, at which the job queue for this client can be found
	char job_shm_names[GLOBAL_JOBS_MAX_JOBS][JOB_MEM_NAME_MAX_LEN];
	pthread_mutex_t requests_q_lock;

} global_jobs_t;

#endif

