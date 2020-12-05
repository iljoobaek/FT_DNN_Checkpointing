/**
Fault Tolerance Structs and function declarations
*/


#include <unordered_map>
#include <stddef.h>
#include <stdlib.h>
#include <sys/types.h> 			// off_t
#include <sys/mman.h>
#include <sys/stat.h>           // for mode constants
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>              // for O_ constants, such as "O_RDWR"
#include <string.h>				// memset(), strncpy
#include <stdio.h>				// *printf, perror
#include <stdint.h>				// int64_t
#include <semaphore.h>			// sem_t, sem_*()
#include <iostream>
#include <vector>
#include <queue>

#include "mid_structs.h"
#include "mid_common.h"

typedef struct {
    job_t * ft_job;
    uint64_t job_inactive_counter;
    uint64_t job_executing_counter;
    uint64_t fault_detection_counter_value;
} replica_job;

typedef struct {
    bool valid;
    char *contents;
} shm_state;

class FT_CARSS {

private:

std::unordered_map<std::string, replica_job> replica_job_map;

public:

// Allocate job as a replica job
void add_replica_job(job_t * job);

// Update the replica jobs to indicate whether main has died or not.
void update_job_executing_status(job_t * job);

// Update job that is now completed and reset the counters
void update_job_completed_status(job_t * job);

// Returns whether any replica job is ready to execute
bool replica_job_ready_to_execute();

// Insert replica jobs, if any, to fifo queue.
void insert_replica_jobs_to_scheduling_queues(std::queue<job_t *> &fifo_jobs, std::vector<job_t *> &executing_jobs, std::queue<job_t*> &completed_jobs);

};


class FT_CARSS_SHM {
    private:
        void* shm_region;
        int shm_size;
        bool shared_mem_valid;
        std::string shared_mem_name;
    public:
        FT_CARSS_SHM(char * shm_name, int size, bool create = false);
        void update_shared_state(char * state);
        char * read_shared_state();
        bool is_shm_valid();
        void destroy();
};

#ifdef __cplusplus
extern "C" {
#endif 
void ft_carss_shm_destroy(void * ft_carss_shm_obj);
void *ft_carss_shm_create(char * shm_name, int size, bool create);
void ft_carss_shm_update_shared_state(void * ft_carss_shm_obj, char * state);
char * ft_carss_shm_read_shared_state(void * ft_carss_shm_obj);
bool ft_carss_shm_is_shm_valid(void * ft_carss_shm_obj);
#ifdef __cplusplus
}
#endif