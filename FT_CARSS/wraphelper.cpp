
static global_init_flag = false;
static global_jobs_t *global_jobs = NULL;

global_jobs_t *__get_global_struct()
{
	if (global_init_flag)
	{
		return global_jobs;
	}
	else 
	{
		// open shared memory
		errno = 0;              //  automatically set when error occurs
		int shm_fd = shm_open(name, O_RDWR, 0660);
		if (shm_fd == -1){
			perror("[Error] in __get_global_struct: shm_open");
			return NULL;
		}

		if (ftruncate(shm_fd, sizeof(global_jobs_t)) != 0) {
			perror("ftruncate");
		}

		// Map pthread mutex into the global memory.
		void *addr = mmap(  NULL,
							sizeof(global_jobs_t),
							PROT_READ|PROT_WRITE,
							MAP_SHARED,
							shm_fd,
							0);

		if (addr == MAP_FAILED) {
			perror("[Error] in __get_global_struct: mmap");
			return NULL;
		}
		// Save mmap'd pointer to global_jobs global var
		global_jobs  = (global_jobs_t *)addr;
		global_init_flag = true;
		printf("Client address! - %p\n", global_jobs);
		return global_jobs;
	}
}

//
// fill in all the job attributes, and return a job_t struct;
job_t *create_job(pid_t pid, int job_id, char *task_name, \
                launch_params_t launch_params,  kernel_params_t kernel_params, void *addr)
{
    job_t *job = (job_t *)addr;
    if (job == NULL){
        perror("[Error] in create_job: malloc returns NULL\n");
        return NULL;
    }
    job->next = NULL;
    job->job_id = job_id;
    job->pid = pid;
    job->task_name = task_name;
    job->launch_params = launch_params;
    job->kernel_params = kernel_params;
    return job;
}

void init_str_mems(int thread_count, int block_count, int arg_count, char* task_name, int data_size, int buf_no, ...)
{
    // get global structure
    global_jobs_t *g_jobs = __get_global_struct();

    // create data
    void *data_ptr;
    va_list ap;
    va_start(ap, buf_no);
    for (int i = 0; i < buf_no; i++)
    {
		char name[10];
		if (i == 0)
			strcpy(name, "a");
		if (i == 1)
			strcpy(name, "b");
		data_ptr = (void *)va_arg(ap, float *);
		data_mem(name, data_ptr, data_size);
    }
    va_end(ap);

    // create job
    pid_t pid = getpid();
    int job_id = 1; 

    launch_params_t launch_p = {.num_blocks = block_count, .num_threads = thread_count};

    DATA_MEM_SIZE = data_size;

    kernel_params_t kernel_p = {.num_args = arg_count, .data_list = (void*)data_ptr};

//     - submit job to the jobs_submitted list, increment total_count;
//     - constantly checking on jobs_completed list to see if my job is finished; (job id list)
//         if yes, go and retrieve the results
    job_t *new_job = create_job(pid, job_id, task_name, launch_p, kernel_p, addr);
    printf("[INFO] -- created a new job; my pid = %d\n", pid);
    enqueue_jobs_submitted(g_jobs, new_job);
    printf("[INFO] -- submitted the job to global queue\n");

    printf("AFTER g_jobs->is_empty = %d, total_count = %d\n", 
                    g_jobs->is_empty, g_jobs-> total_count);
}

