#ifndef QUEUE_H
#define QUEUE_H

#include <stdlib.h>
#include <stdio.h>
#include "mid_structs.h"

// Constants
#define JOB_QUEUE_LENGTH 10

#ifdef __cplusplus
extern "C" {
#endif
// ----------------------------- Client Functions ------------------------------
// submit a job to the global jobs_submitted queue; update total_count, is_empty,
// the jobs_list itself, 
void submit_job(global_jobs_t *gj, job_t *new_job, char *job_shm_name);
// ---------------------------- Server Functions -------------------------------
int peek_job_queued_at_i(global_jobs_t *gb_jobs, job_t **qd_job, int i);
#ifdef __cplusplus
}
#endif

#endif

