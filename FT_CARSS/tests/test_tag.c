#include <stdio.h>		// *fprintf
#include <stdint.h> 	// uint64_t
#include <unistd.h> 	// sleep
#include "mid_structs.h" // global_jobs_t, job_t
#include "tag_gpu.h"	// tag_begin, tag_end

int main(int argc, char **argv) {
	fprintf(stdout, "Testing server+tag lib ...\n");

	const char *task_name = "test_tag_server";
	tag_begin(1, task_name,
			0L, 0L, 0L,
			0.0, 0.0, 0.0,
			0, 0);

	/* Client's 'work' */
	fprintf(stdout, "Client doing work ...\n");
	sleep(2);

	/* Tag_end routine */
	tag_end(1, task_name);
	fprintf(stdout, "Client exitting ...\n");

	return 0;
}

