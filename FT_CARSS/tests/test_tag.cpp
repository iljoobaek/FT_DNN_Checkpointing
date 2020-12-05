/* Usage of Decorator */
#include <unistd.h> // syscall
#include <sys/syscall.h> // SYS_gettid
#include "tag_gpu.h" // tag_decorator()
int fn(int a) { return a + 1; }

int main() 
{
	pid_t tid = (pid_t) syscall (SYS_gettid);

	auto wrapped_fn = tag_decorator(fn, tid, "test");
	int res = wrapped_fn(3);
	std::cout << "Function returned: " << res << std::endl;
}

