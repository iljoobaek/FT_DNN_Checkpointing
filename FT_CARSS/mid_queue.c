/* 
 * 18980 - Middleware
 *      Middleware that follows server-client architecture, and handles clients'
 #      GPU requests by launching CUDA streams for them
 * Elim Zhang 
 * Nov. 27, 2018
 * ---------------------------------------------------------------
 * 
 * To compile: gcc -Wall -Werror -o queue queue.c 
 *
*/

#include <string.h>
#include <assert.h>
#include <sys/mman.h> // munmap
#include "mid_common.h"
#include "mid_structs.h"
#include "mid_queue.h"

// ------------------------------ Client Functions ------------------------------
// visible to client: 
// submit a job to the global jobs_queued queue; update total_count, 
// the jobs_list itself, 
void submit_job(global_jobs_t *gj, job_t *new_job, char *job_shm_name){

	// First, grab the requests_q_lock to modify job_name list
	pthread_mutex_lock(&(gj->requests_q_lock));
	
	// Then modify job_shm_names 
    strncpy(gj->job_shm_names[gj->total_count], job_shm_name,
		   JOB_MEM_NAME_MAX_LEN);    
    gj->total_count += 1;

	// Lastly, unlock
	pthread_mutex_unlock(&(gj->requests_q_lock));
    return;
}

// ---------------------------- Server Functions -------------------------------
/* Not thread-safe by itself */
int peek_job_queued_at_i(global_jobs_t *gj, job_t **qd_job, int i)
{
	assert(gj != NULL);
	assert(gj->total_count > 0);
	assert(gj->total_count > i);

	// Grab the name from job_shm_names 
	char *job_mem_name = gj->job_shm_names[i];
	
	int res = get_shared_job(job_mem_name, qd_job);
	return res;
}
