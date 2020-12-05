#ifndef TAG_GPU_H
#define TAG_GPU_H

#include <stdint.h> // uint64_t
#include <sys/types.h> // pid_t
#include "mid_structs.h" // MAX_NAME_LENGTH

/* Meta-data structs that describe stats for a repeated job_t struct */
typedef struct meta_job {
	// job_t metadata
    pid_t tid;                      // tid (since client may be multithreaded)

	char job_name[MAX_NAME_LENGTH];
    uint64_t last_peak_mem;			// In bytes
	uint64_t avg_peak_mem;
	uint64_t worst_peak_mem;
    int64_t last_exec_time; // Execution time in (us)
	double avg_exec_time;
    int64_t worst_exec_time;
    int64_t best_exec_time;
	unsigned int run_count;

	// running metadata fields
	uint64_t c_mem_util;
	int64_t job_start_us;
} meta_job_t;

/* C-exposed Interface for easier manipulation/creation of a meta_job */
#ifdef __cplusplus
extern "C" {
#endif
void *CreateMetaJob(pid_t tid, const char *job_name,
		uint64_t lpm, uint64_t apm, uint64_t wpm,
		int64_t let, double aet, int64_t wet, int64_t bet,
		unsigned int run_count);
void DestroyMetaJob(void *mj);
#ifdef __cplusplus
}
#endif


/*
 * In order to communicate GPU execution metadata and intent to server,
 * we wrap any amount of work on the GPU (i.e. a function) with "raw"
 * tag_job_begin()/tag_job_end() interface:
 *
 * tag_job_begin() tells server metadata about intended GPU execution, blocks
 ** until the server allows execution (may be sharing the GPU device)
 * tag_job_end() is called after client work is completed to notify server
 ** that client releases GPU resources.
 *
 * See tag_decorator() wrapper function in tag_dec.h to use stateful wrapping of
 * repeated work-unit execution - wrapper will track metadata for subsequent
 * tag_job_*() calls for the same work.
*/

#ifdef __cplusplus
extern "C" {
#endif
int tag_job_begin(pid_t pid, pid_t tid, const char* job_name,
		int64_t slacktime, bool first_flag, bool shareable_flag,
		uint64_t required_mem);

int tag_job_end(pid_t pid, pid_t tid, const char *job_name);
#ifdef __cplusplus
}
#endif

#endif /* TAG_GPU_H */
