/*
 * Simplified mid.cpp
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>           // for mode constants
#include <unistd.h>
#include <fcntl.h>              // for O_ constants, such as "O_RDWR"
#include <signal.h>             // sigint can trigger the clean up process;
#include <dlfcn.h>                             // dlsym, RTLD_DEFAULT
#include <semaphore.h>			// sem_t, sem_*()

#include <algorithm>	// std::find
#include <cassert>		// assert()
#include <queue>		// std::queue, std::priority_queue, std::vector
#include <vector>			// std::vector
#include <memory>		// std::shared_ptr
#include <unordered_map>	// std::unordered_map
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stddef.h>
#include <sched.h>

#include "mid_structs.h"
#include "mid_queue.h"
#include "mid_common.h"
#include "mid_ft.h"



// Helper define for slacktime threshold check
#define SLACKTIME_THRESHOLD (5*SLEEP_MICROSECONDS)
#define CURR_SLACKTIME_PQ_EMPTY_OFFSET (rel_priority_period_count*SLEEP_MICROSECONDS)
#define WITHIN_SLACKTIME_THRESHOLD(s) \
	(s < (int64_t)CURR_SLACKTIME_PQ_EMPTY_OFFSET + SLACKTIME_THRESHOLD)

#define FAULT_DETECTION_TIME_IN_SECONDS 1  // Number of seconds after which the job is determined to be faulty.

struct CompareJobSlack {
public:
	/* One job is higher priority than another if its slacktime is <= to others */
    bool operator()(job_t *&j1, job_t *& j2) {
		return j2->slacktime_us <= j1->slacktime_us;
    }
};
struct CompareJobPtr {
	job_t *lhs;
	public: 
		CompareJobPtr(job_t *_lhs) : lhs(_lhs) {}
		bool operator()(job_t *&rhs) {
			return lhs->pid == rhs->pid \
				&& lhs->tid == rhs->tid \
				&& strcmp(lhs->job_name, rhs->job_name) == 0 ;
		}
};

static uint64_t max_gpu_memory_available; // In B
static uint64_t gpu_memory_available;	// In Bytes

static std::priority_queue<job_t *,\
		std::vector<job_t *>, CompareJobSlack> pq_jobs;
static std::queue<job_t*> fifo_jobs;
static std::vector<job_t*> executing_jobs;
static std::queue<job_t*> completed_jobs;
static std::vector<job_t*> replica_jobs;
static std::unordered_map<pid_t, int> running_pid_jobs;	// How many jobs per pid concurrently running on GPU
static int gpu_excl_jobs = 0;	// How many jobs are running on GPU with non-shareable flag
// In order to maintain a relative ordering of old and new jobs on the pq
// (which is ordered by slacktimes, which is relative to an absolute client
// frame deadline), we must keep a counter for periods in which the pq is not
// empty.
static int rel_priority_period_count = 0;


static int GJ_fd;
static global_jobs_t *GJ;

FT_CARSS ft_carss;  // Object of class type FT_CARSS

// To keep track of currently executing job's time. Used for preemption
// incase of faulty job.
static bool ENABLE_FAULT_DETECTION = true;
//static uint64_t current_executing_job_time_counter = 0;
//static uint64_t no_executing_jobs_counter = 0;
//static uint64_t JOB_FAULT_DETECTION_COUNTER = FAULT_DETECTION_TIME_IN_SECONDS * 1000000 / SLEEP_MICROSECONDS;

// -------------------------- Server Side Functions ----------------------------


void destroy_global_jobs(int GJ_fd)
{
	if (GJ_fd >= 0)
	{
		close(GJ_fd);
		shm_unlink(GLOBAL_MEM_NAME);
	}
}

static bool continue_flag = true;
void handle_sigint(int ign)
{
	// Just flip the continue flag
	fprintf(stderr, "Caught SIGINT, not continuing ...\n");
	continue_flag = false;
}

int job_release_gpu(job_t *comp_job) {
	if (!comp_job) {
		fprintf(stderr, "Couldn't release job because of bad pointer!\n");
		return -1;
	}

	// Verify release of memory
	if (gpu_memory_available + comp_job->required_mem_b < gpu_memory_available) {
		// Something's wrong, overflow occurred when releasing memory
		fprintf(stderr, "Overflow occurred when releasing gpu for job with name %s (wpm %lu, avail %lu)\n!",\
				comp_job->job_name, comp_job->required_mem_b, gpu_memory_available);
		return -2;
	}

	// Acquired memory
	int acquired_mem = comp_job->required_mem_b ? comp_job->required_mem_b : max_gpu_memory_available;

	// Verify decrement number of jobs running under job's pid
	auto it = running_pid_jobs.find(comp_job->pid);
	if (it == running_pid_jobs.end()) {
		fprintf(stderr, "Couldn't release job because cannot find is running pid (%d)!\n", comp_job->pid);
		return -2;
	} else {
		if (running_pid_jobs[comp_job->pid]< 1) {
			fprintf(stderr, "Couldn't release job because too running pid jobs for pid (%d)!\n", comp_job->pid);
			return -2;
		}
	}
	if (!comp_job->shareable_flag) {
		if (gpu_excl_jobs < 1) {
			fprintf(stderr, "Couldn't release job because too few gpu_excl_jobs!\n");
			return -2;
		}
		gpu_excl_jobs--;
	}

	// Actually release memory and reduce running_pid_jobs
	gpu_memory_available += acquired_mem;
	running_pid_jobs[comp_job->pid]--;
	// Remove pid from running_pid_jobs if its tid count is 0
	if (running_pid_jobs[comp_job->pid] == 0) {
		running_pid_jobs.erase(it);
	}
	return 0;
}

// Helper function for bookkeeping of allocating gpu resources for job
void alloc_gpu_for_job(job_t *j) {
	if (j->required_mem_b == 0) {
		// Allocate all of gpu
		gpu_memory_available = 0;
	} else {
		gpu_memory_available -= j->required_mem_b;
	}

	auto it = running_pid_jobs.find(j->pid);
	if (it != running_pid_jobs.end()) {
		// Increment number of threads running for job's pid
		running_pid_jobs[j->pid]++;
	} else {
		running_pid_jobs[j->pid] = 1;
	}

	if (!j->shareable_flag) { // TODO
		gpu_excl_jobs++;
	}
	return;
}

/*
 * A job can acquire the gpu under the following conditions:
 * 1) Job's memory requirement fits on memory available for GPU
 * AND 
 * 2) job can appropriately share the GPU with other threads or pids
 * AND
 * 3) job's slacktime below a threshold relative to server period
 * Returns 0 on success, -1 on wait signal, -2 on abort signal for job
 */
int job_acquire_gpu(job_t *j) {
	if (!j) return -2;

	bool can_alloc_mem = false;
	if (j->required_mem_b > max_gpu_memory_available) {
		// Must abort job, can never run on the GPU
		return -2;
	} else {
		if (j->required_mem_b == 0 && gpu_memory_available == max_gpu_memory_available) {
			can_alloc_mem = true;
		} else if (j->required_mem_b < gpu_memory_available) {
			can_alloc_mem = true;
		} else {
			can_alloc_mem = false;
		}
	}
	if (!can_alloc_mem) {
		// Must wait for jobs to free up GPU mem
		return -1;
	}

	bool should_run_now = false;
	if (j->noslack_flag) {
		should_run_now = true;
	} else {
		// Whether job should run now depends on slacktime threshold
		should_run_now = WITHIN_SLACKTIME_THRESHOLD(j->slacktime_us);
	}
	if (!should_run_now) {
		// Must wait for slacktime to be within running threshold
		return -1;
	}


	bool can_run_now = false;
	if (running_pid_jobs.size() == 0) {
		can_run_now = true;
	} else {
		auto it = running_pid_jobs.find(j->pid);
		if (it != running_pid_jobs.end()) {
			// Job shares pid with running job
			if (j->shareable_flag) {
				can_run_now = true;
			} else if (running_pid_jobs.size() == 1 && !j->shareable_flag) {
				can_run_now = true;
			} else {
				// Job is not shareable and multiple pids running, must wait
				can_run_now = false;
			}
		} else {
			// Job is first of its process to run next to other jobs
			if (j->shareable_flag) {
				if (gpu_excl_jobs == 0) {
					// Can run if no other jobs are not-shareable
					can_run_now = true;
				} else {
					can_run_now = false;
				}
			} else {
				// Job can't share gpu with other processes
				can_run_now = false;
			}
		}
	}
	if (!can_run_now) {
		return -1;
	} else {
		alloc_gpu_for_job(j);
		return 0;
	}
}

/* Wake client, instruct to abort job */
int abort_job(job_t *aj) {
	if (!aj) return -1;

	// Set client's execution flag to abort
	aj->client_exec_allowed = false;
	return sem_post(&(aj->client_wake));
}

/* Wake client with ability to run job */
int trigger_job(job_t *tj) {
	if (!tj) return -1;

	// Set client's execution flag to run
	tj->client_exec_allowed = true;
	return sem_post(&(tj->client_wake));
}

int read_failed_jobs(job_t ** job) {
	std::string filename ="/home/rtml/nandhakishore/cuMiddleware/failedjob.csv";
	struct stat buffer;
	if(stat(filename.c_str(), &buffer) == 0) {
		// Quick way to check if file exists.
		// Read file

		(*job) = (job_t *)malloc(sizeof(job_t)); // Potential memory leak handle this
		printf("Found Failed Job File. \n");
		std::ifstream job_data(filename);
		std::string line;
		int count = 0;
		while(std::getline(job_data, line)) {
			// CSV must be stored in format pid, tid, job_name
			std::stringstream linestream(line);
			std::string current_entry;
			count = 0;
			while(std::getline(linestream, current_entry, ',')) {
				switch(count) {
					case 0:
						(*job)->pid = (pid_t) std::stoi(current_entry);
						count++;
						break;
					case 1:
						(*job)->tid = (pid_t) std::stoi(current_entry);
						count++;
						break;
					default:
						strcpy((*job)->job_name, current_entry.c_str());
				}
			}
		}
		(*job)->req_type = COMPLETED;
		
		if(count != 2) {
			// Not enough data during reading of file [file currently being written to]
			free((*job));
			return 0;
		}
		remove(filename.c_str());
		return 1;
	}
	return 0;
}

int main(int argc, char **argv)
{


	(void)argc; (void) argv;
	fprintf(stdout, "Starting up middleware main...\n");

	cpu_set_t my_set;        /* Define your cpu_set bit mask. */
	CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
	CPU_SET(4, &my_set);     /* set the bit that represents core 7. */
	sched_setaffinity(0, sizeof(cpu_set_t), &my_set); /* Set affinity of tihs process to */
                                                  /* the defined mask, i.e. only 7. */

	max_gpu_memory_available = 1<<30; // 1 GB
	gpu_memory_available = max_gpu_memory_available;
	fprintf(stdout, "GPU Memory has %lu bytes available at init.\n", gpu_memory_available);

	int res;
	if ((res=init_global_jobs(&GJ_fd, &GJ, true)) < 0)
	{
		fprintf(stderr, "Failed to init global jobs queue");
		return EXIT_FAILURE;
	}

	// Set up signal handler
	signal(SIGINT, handle_sigint);

	// Init flags controlling when jobs_queued_* can't be emptied becase
	// GPU is at capacity
	bool queued_wait_for_complete = false;

	// Global jobs queue has been initialized
	// Begin waiting for job_shm_names to be enqueued (sample every 250ms)
	while (continue_flag)
	{
		if(ENABLE_FAULT_DETECTION) {
			// Increment execution counter in each scheduling cycle
			if(executing_jobs.size()){
				ft_carss.update_job_executing_status(executing_jobs[0]);
			} else {
				ft_carss.update_job_executing_status(NULL);
			}
			ft_carss.insert_replica_jobs_to_scheduling_queues(fifo_jobs, executing_jobs, completed_jobs);
		}

		// Grab job_shm_names lock before emptying
		pthread_mutex_lock(&(GJ->requests_q_lock));
		int i=0;
		while ((GJ->total_count - i)> 0)
		{
			int res;
			job_t *q_job;
			if ((res=peek_job_queued_at_i(GJ, &q_job, i)) < 0)
			{
				fprintf(stderr, "Failed to peek job at idx %d. Continuing...\n", i);
			}

			// Enqueue job request to right queue
			if (q_job->req_type == QUEUED) {
				if (q_job->noslack_flag) {
					printf("Adding to FIFO queue \n");
					// Check if job is a fault tolerant job
					std::string jobname(q_job->job_name);
					if(ENABLE_FAULT_DETECTION && jobname.compare(0, 4, "[FT]") == 0){
						ft_carss.add_replica_job(q_job);
						printf("GOT FT JOB . Adding to Replica queue\n");
					} else fifo_jobs.push(q_job);
				} else {
					if (rel_priority_period_count) {
						// Locally modify the relative priority of this job
						// compared to older jobs on the pq
						q_job->slacktime_us += rel_priority_period_count*SLEEP_MICROSECONDS;
					}
					pq_jobs.push(q_job);
					// Just updated pq, can try to process next job immediately
				}
			}
			else {
				ft_carss.update_job_completed_status(q_job);
				completed_jobs.push(q_job);
			}

			// Continue onto next job_shm_name to process
			i++;
		}

		// job_t *failed_job;
		// if(read_failed_jobs(&failed_job)) {
		// 	printf("Pushing failed job %s \n", failed_job->job_name);
		// 	completed_jobs.push(failed_job);
		// }

		// Reset total count
		GJ->total_count = 0;
		pthread_mutex_unlock(&(GJ->requests_q_lock));

		/* Handle all completed jobs first to release GPU resources */
		while (completed_jobs.size()) {
			/* Dequeue job from completed */
			job_t *compl_job = completed_jobs.front();
			completed_jobs.pop();

			/* Handle completed jobs */
			// First, get original job from executing queue
			auto it = std::find_if(executing_jobs.begin(),
					executing_jobs.end(),
					CompareJobPtr(compl_job));
			assert(it != executing_jobs.end());
			job_t *orig_job = executing_jobs[it - executing_jobs.begin()];

			// Next, release job and remove from executing queue
			if (job_release_gpu(orig_job) == 0) {

				
				// Reset current executing counter
				// if(ENABLE_FAULT_DETECTION)
				// {
				// 	if(current_executing_job_time_counter >= JOB_FAULT_DETECTION_COUNTER) {
				// 		// Job is faulty. Now push replica job to fifo queue.
				// 		std::string ft_job_name(orig_job->job_name);
				// 		ft_job_name  = "[FT]" + ft_job_name;
				// 		for(int i=0; i<replica_jobs.size(); i++) {
				// 			job_t * j = replica_jobs[i];
				// 			if(strcmp(j->job_name, ft_job_name.c_str()) == 0) {
				// 				fifo_jobs.push(j);
				// 				printf("Pushing FT job %s to FIFO queue. \n", j->job_name);
				// 				replica_jobs.erase(replica_jobs.begin() + i);
				// 				break;
				// 			}
				// 		}
				// 	}
				// 	current_executing_job_time_counter = 0;
				// }
				// Remove job from executing_jobs on successful release
				executing_jobs.erase(it);

				/* TODO: Avoid having client sleep waiting for server */
				// NOTE: It must be the compl_job the client is holding on to
				sem_post(&(compl_job->client_wake));

				// Reset flags since a job just released gpu resources
				queued_wait_for_complete = false;

				// Destroy shared jobs
				destroy_shared_job(&orig_job);
				destroy_shared_job(&compl_job);
			} else  {
				fprintf(stderr, "Something went wrong releasing job's resources!\n");
			}

		}


		/*
		 * Next, run any jobs that have never run yet (and therefore have no priority).
		 */
		while (!queued_wait_for_complete &&
				fifo_jobs.size()) {
			/* Peek at job from jobs_queued */
			job_t *q_job = fifo_jobs.front();

			/* Handle queued jobs */
			int res = job_acquire_gpu(q_job);
			if (res < 0) {
				if (res == -1) {
					// Failed to acquire GPU, must wait for other jobs to complete
					queued_wait_for_complete = true;

					break;
				} else {
					// Job is too big to fit on the GPU, instruct client to abort
					// job
					fprintf(stdout, "\tJob must ABORT!\n");
					abort_job(q_job);
					// Destroy shared job
					destroy_shared_job(&q_job);
				}
			} else {
				// Adds q_job to executing_jobs on success
				executing_jobs.push_back(q_job);

				// Wake client to trigger execution
				fprintf(stdout, "\tJob (%s, pid=%d, tid=%d) can execute!\n", q_job->job_name, q_job->pid, q_job->tid);
				if (trigger_job(q_job) < 0) {
					fprintf(stderr, "\tFailed to wake client!\n");
				}
			}
			// Actually pop job off queue
			fifo_jobs.pop();
		}

		/*
		 * Lastly, run as many jobs (according to priority) as can fit on GPU.
		 */
		if (pq_jobs.size()) {
			// Priority queue is not empty, increment the rel_priority_period_count
			rel_priority_period_count++;
		}
		while (pq_jobs.size()) {
			/* Peek at top job from pq_jobs */
			job_t *q_job = pq_jobs.top();

			/* Handle queued jobs */
			int res = job_acquire_gpu(q_job);
			if (res < 0) {
				if (res == -1) {
					// Failed to acquire GPU, must wait for other jobs to complete
					break;
				} else {
					// Job is too big to fit on the GPU, instruct client to abort
					// job
					fprintf(stdout, "\tJob must ABORT!\n");
					abort_job(q_job);
					// Destroy shared job
					destroy_shared_job(&q_job);
				}
			} else {
				// Adds q_job to executing_jobs on success
				executing_jobs.push_back(q_job);

				// Wake client to trigger execution
				fprintf(stdout, "\tJob (%s, pid=%d, tid=%d) can execute!\n", q_job->job_name, q_job->pid, q_job->tid);
				if (trigger_job(q_job) < 0) {
					fprintf(stderr, "\tFailed to wake client!\n");
				}
			}
			// Pop job off priority-queue 
			pq_jobs.pop();

			if (pq_jobs.size() == 0) {
				// Reset the relative priority period counter
				rel_priority_period_count = 0;
			}
		}

		
		// Sleep before starting loop again
		usleep(SLEEP_MICROSECONDS);
	}

	fprintf(stdout, "Cleaning up server...\n");
	destroy_global_jobs(GJ_fd);
	return 0;
}
