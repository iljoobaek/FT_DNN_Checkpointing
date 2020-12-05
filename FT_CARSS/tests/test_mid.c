#include <stdio.h> // fprintf
#include <pthread.h> // pthread*
#include <stdint.h> // uint64_t
#include <stdbool.h> // bool
#include <time.h> // time_t
#include <sys/mman.h> // shm*
#include <unistd.h>	// sleep

#include "mid_common.h" // init_global_jobs, build_shared_job
#include "mid_queue.h" // submit_job

int main(int argc, char **argv) {
	fprintf(stdout, "Testing server ...\n");

	int fd;
	global_jobs_t *gb_jobs;
	init_global_jobs(&fd, &gb_jobs);

	char job_shm_name[JOB_MEM_NAME_MAX_LEN];
	job_t *tagged_job;
	build_shared_job(1, "test_server",
			0L, 0L, 0L,
			0.0, 0.0, 0.0,
			0, 0,
			QUEUED,
			&tagged_job, job_shm_name);

	fprintf(stdout, "Client submitting job to server\n");
	submit_job(gb_jobs, tagged_job, job_shm_name);

	fprintf(stdout, "Client cond waiting\n");
	pthread_cond_wait(&(tagged_job->client_wake), &(tagged_job->own_job));
	pthread_mutex_unlock(&(tagged_job->own_job));
	fprintf(stdout, "Client thread woke up! Exec flag %d\n",
			tagged_job->client_exec_allowed);
	
	shm_unlink(job_shm_name);

	/* Client's 'work' */
	fprintf(stdout, "Client doing work ...\n");
	sleep(2);

	/* Tag_end routine */
	fprintf(stdout, "Client finished work, communicating to server\n");
	build_shared_job(1, "test_server",
			0L, 0L, 0L,
			0.0, 0.0, 0.0,
			0, 0,
			COMPLETED,
			&tagged_job, job_shm_name);

	fprintf(stdout, "Client submitting COMPLETED job to server\n");
	submit_job(gb_jobs, tagged_job, job_shm_name);

	fprintf(stdout, "Client cond waiting\n");
	pthread_cond_wait(&(tagged_job->client_wake), &(tagged_job->own_job));
	pthread_mutex_unlock(&(tagged_job->own_job));
	fprintf(stdout, "Client thread woke up! Exec flag %d\n",
			tagged_job->client_exec_allowed);
	
	shm_unlink(job_shm_name);

	fprintf(stdout, "Client exitting ...\n");

	return 0;
}
