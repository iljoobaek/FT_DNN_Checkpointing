#include <stdio.h> //fprintf
#include <unistd.h> // sleep()
#include "tag_gpu.h"
int main(int argc, char *argv){

	for (int i = 0; i < 10000; i++)
	{
		fprintf(stdout, "opengl: tag_beginning() %d\n", i);
        // Tagging begin ///////////////////////////////////////////////////////////////////
        const char *task_1_name = "opengl_get_image";
        tag_begin(2, task_1_name, 0L, 0L, 0L, 0.0, 0.0, 0.0, 0, 0);

        usleep(1000);

        // Tag_end /////////////////////////////////////////////////////////////////////////
        tag_end(2, task_1_name);

        // Tagging begin ///////////////////////////////////////////////////////////////////
        const char *task_2_name = "opengl_draw";
        tag_begin(2, task_2_name, 0L, 0L, 0L, 0.0, 0.0, 0.0, 0, 0);

        usleep(1000);

        // Tag_end /////////////////////////////////////////////////////////////////////////
        tag_end(2, task_2_name);

        // Tagging begin ///////////////////////////////////////////////////////////////////
        const char *task_3_name = "opengl_glBegin";
        tag_begin(2, task_3_name, 0L, 0L, 0L, 0.0, 0.0, 0.0, 0, 0);

        usleep(1000);

        // Tag_end /////////////////////////////////////////////////////////////////////////
        tag_end(2, task_3_name);

        // Tagging begin ///////////////////////////////////////////////////////////////////
        const char *task_4_name = "opengl_swapBuffer";
        tag_begin(2, task_4_name, 0L, 0L, 0L, 0.0, 0.0, 0.0, 0, 0);

        usleep(1000);

        // Tag_end /////////////////////////////////////////////////////////////////////////
        tag_end(2, task_4_name);

		fprintf(stdout, "opencv_draw: onto next job in 1s\n");
		usleep(1000);
	}
}
