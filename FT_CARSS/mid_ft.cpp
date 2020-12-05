#include "mid_ft.h"

/*
Extract tokens within [] brackets. Eg: "[FT][500][X]safhdshdjh" => {"FT", "500", "X"}
TODO: Update with Regex if possible
*/
int extract_tokens(std::string str, std::vector<std::string> &results) {
    int i=0;
    int n = str.size();
    while(i < n) {
        if(str[i] == '[') {
            i++;
            std::string current_str;
            while(i < n && str[i] != ']') {
                current_str += str[i];
                i++;
            }
            results.push_back(current_str);
            i++;
        } else break;
    }
    return i;
}

/*
Adds a job to the replica jobs map, if the job name contains [FT] in the title.
*/
void FT_CARSS::add_replica_job(job_t * job) {
    std::string ft_job_name(job->job_name);

    // Extract FT milliseconds from the name.
    std::vector<std::string> tokens;
    int starting_idx = extract_tokens(ft_job_name, tokens);

    std::string actual_job_name(ft_job_name.begin() + starting_idx, ft_job_name.end());

    if(tokens.size() < 2) // Must contain atleast [FT][milliseconds]
        return;

    replica_job current_job = {
        .ft_job = job,
        .job_inactive_counter = 0,
        .job_executing_counter = 0,
        .fault_detection_counter_value = std::stoi(tokens[1]) * 1000 / SLEEP_MICROSECONDS
    };

    replica_job_map[actual_job_name] = current_job;
    std::cout<<"[FTCARSS] Successfully added FT job: " << actual_job_name << " with fault time " << current_job.fault_detection_counter_value << " to monitoring\n";
}

/*
By passing job pointers to this function everytime a job arrives, FT CARSS can
update the appropriate counters for jobs and denote the 
*/
void FT_CARSS::update_job_executing_status(job_t *job) {
    

    for(auto &x : replica_job_map) {
        x.second.job_inactive_counter += 1; // The active job will get its inactive counter reset below.
    }

    if(!job)
        return;

    std::string job_name(job->job_name);

    if(replica_job_map.find(job_name) != replica_job_map.end()) {
        replica_job_map[job_name].job_executing_counter += 1;
        replica_job_map[job_name].job_inactive_counter = 0;
    }
}

/*
Listens to jobs that arrive as completed to reset both counters in the replica.
*/
void FT_CARSS::update_job_completed_status(job_t *job) {
    std::string job_name(job->job_name);
    if(replica_job_map.find(job_name) != replica_job_map.end()) {
        replica_job_map[job_name].job_executing_counter = 0;
        replica_job_map[job_name].job_inactive_counter = 0;
    }
}


/*
Returns true if a job is either executing for too long / inactive for too long
*/
bool FT_CARSS::replica_job_ready_to_execute() {
    for(auto &x : replica_job_map) {
        if(x.second.job_inactive_counter > x.second.fault_detection_counter_value || x.second.job_executing_counter > x.second.fault_detection_counter_value)
            return true;
    }
    return false;
}

/*
Inserts a replica job into fifo queue if:
1. Job executes for too long. This means that GPU is not released. here, job from executing queue is inserted into completed queue to evict executing job.
2. Executing queue inactive for too long.
*/
void FT_CARSS::insert_replica_jobs_to_scheduling_queues(std::queue<job_t *> &fifo_jobs, std::vector<job_t *> &executing_jobs, std::queue<job_t*> &completed_jobs) {
    if(executing_jobs.size() > 0) {
        std::string executing_job_name(executing_jobs[0]->job_name);
        if(replica_job_map.find(executing_job_name) != replica_job_map.end()) {
            if(replica_job_map[executing_job_name].job_executing_counter > replica_job_map[executing_job_name].fault_detection_counter_value) {
                completed_jobs.push(executing_jobs[0]);
                fifo_jobs.push(replica_job_map[executing_job_name].ft_job);
                replica_job_map.erase(executing_job_name);
                std::cout<<"[FT CARSS] Replacing stale executing job " << executing_job_name <<" with completion job\n";
            }
        }
    }
    for(auto it = replica_job_map.begin(); it != replica_job_map.end(); ) {
        if((*it).second.job_inactive_counter > (*it).second.fault_detection_counter_value) {
            fifo_jobs.push((*it).second.ft_job);
            it = replica_job_map.erase(it);
            std::cout<<"[FT CARSS] Replacing inactive main job with replica job. \n";
        } else ++it;
    }
}


FT_CARSS_SHM::FT_CARSS_SHM(char * shm_name, int size, bool create) {
    std::vector<std::string> tokens;
    std::string shm_name_string_dirty(shm_name);
    int starting_idx = extract_tokens(shm_name_string_dirty, tokens);
    std::string shm_name_string(shm_name_string_dirty.begin() + starting_idx, shm_name_string_dirty.end());
    int fd = shm_open(shm_name_string.c_str(), O_RDWR, 0660);
    if(errno == ENOENT) {
        if(create) {
            fd = shm_open(shm_name, O_CREAT|O_RDWR, 0660);
            if(ftruncate(fd, size) != 0) {
                perror("ftruncate - ftcarssshm");
            }
        } else {
            shared_mem_valid = false;
        }
    }
    if(fd == -1)
        return;
    shared_mem_valid = true;
    shm_region = (void *)mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    shared_mem_name = shm_name_string;
    if(shm_region == MAP_FAILED)
        return;
    shm_size = size;
    std::cout<<" Created shared state size : " << shm_size <<" Name: " << shared_mem_name << "\n";
}

void FT_CARSS_SHM::update_shared_state(char * state) {
    int size = strlen(state);
    std::cout<<"Update frame state: " << std::string(state) << "\n";
    memcpy(shm_region, state, size+1);
}

bool FT_CARSS_SHM::is_shm_valid() {
    return shared_mem_valid;
}

char * FT_CARSS_SHM::read_shared_state() {
    // Possible mem leak here. Since this is only called once, this is kind-of acceptable I guess

    char * return_string = (char *)malloc(sizeof(char) * shm_size);
    memcpy(return_string, shm_region, shm_size);
    std::cout<<"Read shared state: " << std::string(return_string) << " \n";
    return return_string;
}

void FT_CARSS_SHM::destroy() {
    std::cout<<" Destroyed shared state \n";
    shm_unlink(shared_mem_name.c_str());
}



extern "C" 
{
    void ft_carss_shm_destroy(void * ft_carss_shm_obj){
        FT_CARSS_SHM *fcshm  = reinterpret_cast<FT_CARSS_SHM *>(ft_carss_shm_obj);
        fcshm->destroy();
        delete fcshm;
    }

    void *ft_carss_shm_create(char * shm_name, int size, bool create) {
        return reinterpret_cast<void *> (new FT_CARSS_SHM(shm_name, size, create));
    }

    void ft_carss_shm_update_shared_state(void * ft_carss_shm_obj, char * state) {
        FT_CARSS_SHM *fcshm  = reinterpret_cast<FT_CARSS_SHM *>(ft_carss_shm_obj);
        fcshm->update_shared_state(state);
    }

    char * ft_carss_shm_read_shared_state(void * ft_carss_shm_obj) {
        FT_CARSS_SHM *fcshm  = reinterpret_cast<FT_CARSS_SHM *>(ft_carss_shm_obj);
        return fcshm->read_shared_state();
    }

    bool ft_carss_shm_is_shm_valid(void * ft_carss_shm_obj) {
        FT_CARSS_SHM *fcshm  = reinterpret_cast<FT_CARSS_SHM *>(ft_carss_shm_obj);
        return fcshm->is_shm_valid();
    }
}