#include <sys/types.h>  // pid_t
#include "mid_structs.h" // job_t

extern "C"
int queue_job_to_server(pid_t pid, const char *fn_name, job_t *global_jobs_q);

