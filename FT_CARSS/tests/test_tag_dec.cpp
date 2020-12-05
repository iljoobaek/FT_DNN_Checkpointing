#include <iostream>
#include <sys/types.h> /* pid_t */

#include "tag_dec.h"
#include "tag_gpu.h" // meta_job_t


int modify_metrics() {
	pid_t tid = gettid();
	meta_job_t *rmj = get_running_meta_job_for_tid(tid);

	rmj->c_mem_util = 100;
	return 0;
}

int foo() {
	modify_metrics();
	return 0;
}

int main(int argc, char **argv) {
	meta_job_t foo_meta = meta_job_t();
	const char *name = "testing_foo";
	std::strcpy(foo_meta.job_name, name);
	foo_meta.c_mem_util = 0;
	std::cout << "foo has wpm: " << foo_meta.worst_peak_mem<< "\n" << std::endl;

	auto dec_foo_ptr = tag_decorator(foo, (const meta_job_t*)&foo_meta);

	std::cout << "Calling decorated foo() ...\n" << std::endl;
	std::cout << (*dec_foo_ptr)() << "\n" << std::endl;


	// Only one main thread, get the modified metadata job 
	pid_t tid = gettid();
	meta_job_t *rmj = get_running_meta_job_for_tid(tid);
	std::cout << "Mem util is " << rmj->c_mem_util << "\n" << std::endl;
	return 0;
}
