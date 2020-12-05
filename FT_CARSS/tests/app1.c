#include <stdio.h> //fprintf
#include <unistd.h> // sleep()
#include <sys/types.h> // pid_t
#include "tag_frame.h" // FrameController
#include "tag_gpu2.h"

int main(int argc, char **argv) {

	pid_t pid = 1;
	pid_t tid = 0;
	const char *job_name = "opencv_get_image";
	for (int i = 0; i < 10000; i++)
	{
		fprintf(stdout, "opencv: tag_beginning() %d\n", i);
		// Tagging begin ///////////////////////////////////////////////////////////////////
		tag_job_begin(pid, tid, job_name, 
				15L, true, true, 1UL);

        usleep(1000);

		// Tag_end /////////////////////////////////////////////////////////////////////////
		tag_job_end(pid, tid, job_name);

		fprintf(stdout, "opencv_get_image: onto next job in 1s\n");
		usleep(1000);
	}
}
