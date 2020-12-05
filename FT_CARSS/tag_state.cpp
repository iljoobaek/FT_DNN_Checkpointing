#include <cstdio> /* fprintf */
#include <cstring> /* memcpy */
#include <iostream> /* std::endl, std::cout */
#include <sys/types.h> /* pid_t */
#include <stdint.h> // int64_t
#include <mutex> /* std::unique_lock, std::mutex */
#include <unordered_map> // unordered_map
#include <unistd.h>  // getpid()
#include <chrono>	// std::chrono::high_resolution_clock

#include "mid_structs.h" // MAX_NAME_LENGTH
#include "tag_state.h" // *_running_meta_job_for_tid, TagState class
#include "tag_gpu.h" /* meta_job_t */

static std::mutex hm_lock;
static std::unordered_map<pid_t, meta_job_t *> g_tid_curr_job;

/* Interface for getting meta_job_t from g_tid_curr_job */
#ifdef __cplusplus
extern "C" {
#endif 
meta_job_t *get_running_meta_job_for_tid(pid_t tid) {
	// Use unique_lock to manage hashmap lock
	std::unique_lock<std::mutex> lck;
	lck = std::unique_lock<std::mutex>(hm_lock); // Locks on construction

	if (g_tid_curr_job.find(tid) == g_tid_curr_job.end()) {
		return NULL;
	} else {
		return g_tid_curr_job[tid];
	}
	// Unlocks mutex on destruction
}

int set_running_job_for_tid(pid_t tid, meta_job_t *mj) {
	// Use unique_lock to manage hashmap lock
	std::unique_lock<std::mutex> lck;
	lck = std::unique_lock<std::mutex>(hm_lock); // Locks on construction

	g_tid_curr_job[tid] = mj;
	return 0;
	// Unlocks on destruction
}

#ifdef __cplusplus
}
#endif

/* Define TagState class implementation */
	/* Default constructor */
TagState::TagState() : init_meta_job(std::make_shared<meta_job_t>()) {
	memset(init_meta_job.get(), 0, sizeof(meta_job_t));
	// Save process' pid for use in tag_job_* calls
	tag_pid = getpid();
}

TagState::TagState(const char *job_name) : init_meta_job(std::make_shared<meta_job_t>()) {
	// empty bits in init_meta_job, save given job name
	memset(init_meta_job.get(), 0, sizeof(meta_job_t));
	strncpy(init_meta_job->job_name, job_name, MAX_NAME_LENGTH);
	init_meta_job->tid = 0;
	// Save process' pid for use in tag_job_* calls
	tag_pid = getpid();
}

TagState::TagState(const meta_job_t *inp_init_meta_job) : init_meta_job(std::make_shared<meta_job_t>()) {
	/* Init a template job for all threads calling this function */
	memcpy(init_meta_job.get(), inp_init_meta_job, sizeof(meta_job_t));
	init_meta_job->tid = 0;

	// Save process' pid for use in tag_job_* calls
	tag_pid = getpid();
	}

TagState::~TagState() { 
	tid_to_meta_job.clear();
}

/* Retrieve locally hashed current job metadata template:
 * If tid has no entry in hashmap, 
 * 	one is created copied from init_meta_job 
 * Else, return pointer to stored in hashmap for tid
 */
meta_job_t *TagState::get_local_meta_job_for_tid(pid_t tid) {
	// Use unique_lock to manage hashmap lock
	std::unique_lock<std::mutex> lck;
	lck = std::unique_lock<std::mutex>(hm_lock); // Locks on construction

	if (tid_to_meta_job.find(tid) == tid_to_meta_job.end()) {
		// No meta_job entry present, must create new one 
		// as a copy of init_meta_job
		std::shared_ptr<meta_job_t> new_mj =\
			std::shared_ptr<meta_job_t>(new meta_job_t);
		memcpy(new_mj.get(), init_meta_job.get(), sizeof(meta_job_t));

		// Tweak a few parameters of new_mj
		new_mj->tid = tid;

		tid_to_meta_job.insert(std::make_pair(tid, new_mj));
	}

	// Return the stored pointer in shared_ptr saved in hashmap
	return tid_to_meta_job[tid].get();
	// Unlocks on destruction of lck
}

int TagState::acquire_gpu(pid_t call_tid, int64_t slacktime, bool first, bool shareable) {
	// First, retrieve tid and local copy of current metadata for job
	meta_job_t *curr_meta_job = get_local_meta_job_for_tid(call_tid);

	// Set the global hashmap of tid->job metadata to point
	// to this thread's job (i.e. for this tid)
	set_running_job_for_tid(call_tid, curr_meta_job);

	int res = tag_job_begin(tag_pid, call_tid, curr_meta_job->job_name, slacktime, first, shareable,
			curr_meta_job->worst_peak_mem);

	// Get current time in microseconds
	std::chrono::system_clock::time_point now_tp = std::chrono::high_resolution_clock::now();
	std::chrono::microseconds us_now = std::chrono::duration_cast<std::chrono::microseconds>(now_tp.time_since_epoch());
	curr_meta_job->job_start_us = us_now.count();
	return res;
}

int TagState::release_gpu(pid_t call_tid) {
	// First, retrieve tid and local copy of current metadata for job
	meta_job_t *curr_meta_job = get_local_meta_job_for_tid(call_tid);

	// Notify server of end of work
	int res = tag_job_end(tag_pid, call_tid, curr_meta_job->job_name);

	// Update tid's job execution time stats:
	// last_exec_time, worst_exec_time, avg_exec_time, best_exec_time
	std::chrono::system_clock::time_point now_tp = std::chrono::high_resolution_clock::now();
	std::chrono::microseconds us_now = std::chrono::duration_cast<std::chrono::microseconds>(now_tp.time_since_epoch());
	int64_t job_end_us = us_now.count();
	int64_t job_time_us = job_end_us - curr_meta_job->job_start_us;
	unsigned int rc = curr_meta_job->run_count;
	// Last execution time
	curr_meta_job->last_exec_time = job_time_us;
	// Worst execution time
	if (!rc || job_time_us > curr_meta_job->worst_exec_time) {
		curr_meta_job->worst_exec_time = job_time_us;
	}
	// Best execution time
	if (!rc || job_time_us < curr_meta_job->best_exec_time) {
		curr_meta_job->best_exec_time = job_time_us;
	}
	// Update avg execution time
	double c_avg_exec_time = curr_meta_job->avg_exec_time;
	curr_meta_job->avg_exec_time = \
		   (c_avg_exec_time * rc + (double)(job_time_us))/(rc+1);

	// TODO: Actually collect memory metrics intercepted from CUDA calls
	curr_meta_job->worst_peak_mem = 1LU;

	// Update the runcount
	curr_meta_job->run_count++;
	return res;
}

/* Stats retrieval */
// All return -1 if not yet known
int64_t TagState::get_wc_exec_time_for_tid(pid_t tid) const   {
	auto it = tid_to_meta_job.find(tid);
	if (it == tid_to_meta_job.end()) {
		return -1;
	} else if (tid_to_meta_job.at(tid)->worst_exec_time == 0) {
		return -1;
	}
	return tid_to_meta_job.at(tid)->worst_exec_time;
}
int64_t TagState::get_max_wc_exec_time() const {
	// Loop over each thread's meta_jobs and find maximum
	int64_t max_wc_exec_time = -1;
	for (auto iter = tid_to_meta_job.begin(); iter != tid_to_meta_job.end(); ++iter) {
		std::shared_ptr<meta_job_t> mj = iter->second;
		if (mj->worst_exec_time > max_wc_exec_time)	 {
			max_wc_exec_time = mj->worst_exec_time;
		}
	}
	return max_wc_exec_time;
}

// Returns -1 if not yet known
int64_t TagState::get_required_mem_for_tid(pid_t tid) const {
	auto it = tid_to_meta_job.find(tid);
	if (it == tid_to_meta_job.end()) {
		return -1;
	} else if (tid_to_meta_job.at(tid)->worst_peak_mem == 0) {
		return -1;
	}
	return tid_to_meta_job.at(tid)->worst_peak_mem;
}

int64_t TagState::get_best_exec_time_for_tid(pid_t tid) const {
	auto it = tid_to_meta_job.find(tid);
	if (it == tid_to_meta_job.end()) {
		return -1;
	} else if (tid_to_meta_job.at(tid)->best_exec_time == 0) {
		return -1;
	}
	return tid_to_meta_job.at(tid)->best_exec_time;
}

int64_t TagState::get_overall_best_exec_time() const {
	int64_t bet = -1;
	for (auto iter = tid_to_meta_job.begin(); iter != tid_to_meta_job.end(); ++iter) {
		std::shared_ptr<meta_job_t> mj = iter->second;
		if (bet == -1 || mj->best_exec_time < bet) {
			bet = mj->best_exec_time;
		}
	}
	return bet;
}

int64_t TagState::get_worst_exec_time_for_tid(pid_t tid) const {
	auto it = tid_to_meta_job.find(tid);
	if (it == tid_to_meta_job.end()) {
		return -1;
	} else if (tid_to_meta_job.at(tid)->worst_exec_time == 0) {
		return -1;
	}
	return tid_to_meta_job.at(tid)->worst_exec_time;
}

int64_t TagState::get_overall_worst_exec_time() const {
	int64_t wet = -1;
	for (auto iter = tid_to_meta_job.begin(); iter != tid_to_meta_job.end(); ++iter) {
		std::shared_ptr<meta_job_t> mj = iter->second;
		if (wet == -1 || mj->worst_exec_time > wet) {
			wet = mj->worst_exec_time;
		}
	}
	return wet;
}

int64_t TagState::get_last_exec_time_for_tid(pid_t tid) const {
	auto it = tid_to_meta_job.find(tid);
	if (it == tid_to_meta_job.end()) {
		return -1;
	} else if (tid_to_meta_job.at(tid)->last_exec_time == 0) {
		return -1;
	}
	return tid_to_meta_job.at(tid)->last_exec_time;
}

int64_t TagState::get_worst_last_exec_time() const {
	// Loop over all threads' last exec time, return worst
	int64_t ret = -1;
	for (auto iter = tid_to_meta_job.begin(); iter != tid_to_meta_job.end(); ++iter) {
		std::shared_ptr<meta_job_t> mj = iter->second;
		if (ret == -1 || ret < mj->last_exec_time) {
			ret = mj->last_exec_time;
		}
	}
	return ret;
}

double TagState::get_avg_exec_time_for_tid(pid_t tid) const {
	auto it = tid_to_meta_job.find(tid);
	if (it == tid_to_meta_job.end()) {
		return -1;
	} else if (tid_to_meta_job.at(tid)->avg_exec_time == 0.0) {
		return -1;
	}
	return tid_to_meta_job.at(tid)->avg_exec_time;
}

double TagState::get_overall_avg_exec_time() const {
	// Average all tid's avg_exec_times (weighted by runcounts)
	double aet = -1.0;
	int tot_count = 0;
	for (auto iter = tid_to_meta_job.begin(); iter != tid_to_meta_job.end(); ++iter) {
		std::shared_ptr<meta_job_t> mj = iter->second;
		if (mj->avg_exec_time>= 0.0) {
			if (aet == -1) {
				// Initialize aet before aggregation 
				aet = 0.0;
			}
			aet += mj->avg_exec_time * mj->run_count;
		}
		tot_count+=mj->run_count;
	}
	aet = aet / (double)tot_count;
	return aet;
}

// Print execution timing stats for all threads
void TagState::print_exec_stats() {
	for (auto iter = tid_to_meta_job.begin(); iter != tid_to_meta_job.end(); ++iter) {
		std::shared_ptr<meta_job_t> mj = iter->second;
		std::cout << "EXEC_STATS: Tagged work (" << mj->job_name\
			<< ", tid:" << mj->tid << ") -> "\
			<< mj->best_exec_time << "us, " << mj->avg_exec_time << " us, "\
			<< mj->worst_exec_time << "us (min, avg, max) over " << mj->run_count\
			<< " runs for this thread." << std::endl;
	}
}


/* Implement the C-exposed interface for TagState objects */
#ifdef __cplusplus
extern "C" {
#endif 
void *CreateTagStateObj(const meta_job_t *inp_init_meta_job) {
	return reinterpret_cast<void *>(new TagState(inp_init_meta_job));
}
void DestroyTagStateObj(void *tag_obj) {
	delete reinterpret_cast<TagState *>(tag_obj);
}
meta_job_t *TagState_get_local_meta_job_for_tid(void *tag_obj, pid_t tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_local_meta_job_for_tid(tid);
}
int TagState_acquire_gpu(void *tag_obj, pid_t tid,
	   	int64_t slacktime, bool first_flag, bool shareable_flag) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->acquire_gpu(tid, slacktime, first_flag, shareable_flag);
}
int TagState_release_gpu(void *tag_obj, pid_t call_tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->release_gpu(call_tid);
}
int64_t TagState_get_wc_exec_time_for_tid(void *tag_obj, pid_t tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_wc_exec_time_for_tid(tid);
}
int64_t TagState_get_max_wc_exec_time(void *tag_obj) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_max_wc_exec_time();
}
uint64_t TagState_get_required_mem_for_tid(void *tag_obj, pid_t tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_required_mem_for_tid(tid);
}
int64_t TagState_get_best_exec_time_for_tid(void *tag_obj, pid_t tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_best_exec_time_for_tid(tid);
}
int64_t TagState_get_worst_exec_time_for_tid(void *tag_obj, pid_t tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_worst_exec_time_for_tid(tid);
}
int64_t TagState_get_last_exec_time_for_tid(void *tag_obj, pid_t tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_last_exec_time_for_tid(tid);
}
int64_t TagState_get_worst_last_exec_time(void *tag_obj) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_worst_last_exec_time();
}
double TagState_get_avg_exec_time_for_tid(void *tag_obj, pid_t tid) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_avg_exec_time_for_tid(tid);
}
int64_t TagState_get_overall_best_exec_time(void *tag_obj) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_overall_best_exec_time();
}
int64_t TagState_get_overall_worst_exec_time(void *tag_obj) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_overall_worst_exec_time();
}
double TagState_get_overall_avg_exec_time(void *tag_obj) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->get_overall_avg_exec_time();
}
void TagState_print_exec_stats(void *tag_obj) {
	TagState *ts = reinterpret_cast<TagState *>(tag_obj);
	return ts->print_exec_stats();
}
#ifdef __cplusplus
}
#endif 

