#include <stdio.h> //fprintf
#include <unistd.h> // sleep()
#include <sys/types.h> // pid_t
#include "tag_frame.h" // FrameController
#include "tag_state.h" // gettid()
#include "tag_dec.h" // frame_job_decorator
#include "tag_gpu.h"

// Work functions must NOT be of return-type void
int app4_work1() {
	fprintf(stderr, "Work1\n");
	usleep(10000); // 10ms
	return 0;
}

int app4_work2() {
	fprintf(stderr, "Work2\n");
	usleep(10000);
	return 0;
}

int main(int argc, char **argv) {

	pid_t pid = getpid();
	pid_t tid = gettid();
	const char *frame_name = "app4";

	FrameController fc(frame_name, 50, false);

	// decorate work functions
	auto tagged_work1_ptr = frame_job_decorator(app4_work1, &fc, "app4_work1", false);
	auto tagged_work2_ptr = frame_job_decorator(app4_work2, &fc, "app4_work2", false);
	for (int i = 0; i < 10; i++)
	{
		fc.frame_start();
		try {
			fprintf(stdout, "opencv: tag_beginning() %d\n", i);

			(*tagged_work1_ptr)();
			(*tagged_work2_ptr)();
		} catch (DroppedFrameException& e) {
			// Can continue in the while loop, just skipping this frame due to
			// recoverable abort of frame-jobs.
			fprintf(stderr, "opencv: had to drop a frame! Continuing...\n");
		}
		fc.frame_end();
	}
	fc.print_exec_stats();
}
