#include <cstdio> /* fprintf */
#include <cstring> /* memcpy */
#include <deque> /* std::deque */
#include <iostream> /* std::cout, std::endl */
#include <sys/types.h> /* pid_t */
#include <mutex> /* std::unique_lock, std::mutex */
#include <chrono> /* std::chrono::high_resolution_clock::* */
#include <stdexcept> /* std::exception */
#include <stdint.h> // int64_t
#include <cmath> // round(double)

#include "mid_structs.h" // MAX_NAME_LENGTH
#include "tag_state.h" // TagState
#include "tag_frame.h" // Implements fn's in this header file

/* Define FrameJob class implementation */
FrameJob::FrameJob(const char* _job_name, bool shareable_flag)
	:  job_name(_job_name), job_state(_job_name), \
		shareable(shareable_flag), wc_exec_time(-1),
	wc_remaining_exec_time(-1) {
}

FrameJob::~FrameJob() {}
// Return expected execution time for thread using FrameJob's job_state
int64_t FrameJob::get_expected_exec_time_for_tid(pid_t tid) {
	return job_state.get_wc_exec_time_for_tid(tid);
}
uint64_t FrameJob::get_expected_required_mem_for_tid(pid_t tid) {
	return job_state.get_required_mem_for_tid(tid);
}
int64_t FrameJob::get_wc_remaining_exec_time() {
	return wc_remaining_exec_time;
}
void FrameJob::update_wc_exec_time() {
	wc_exec_time = job_state.get_max_wc_exec_time();
}
void FrameJob::set_wc_remaining_exec_time(int64_t rem_exec_time) {
	wc_remaining_exec_time = rem_exec_time;
}
int FrameJob::prepare_job(pid_t tid, int64_t slacktime, bool first_flag) {
	int res = job_state.acquire_gpu(tid, slacktime, first_flag, shareable);
	if (res < 0) {
		fprintf(stderr, "Could not prepare job (%s)!\n", job_name.c_str());
	}
	return res;
}
int FrameJob::release_job(pid_t tid) {
	int res = job_state.release_gpu(tid);
	return res;
}

int64_t FrameJob::get_worst_last_exec_time() const {
	return job_state.get_worst_last_exec_time();
}

int64_t FrameJob::get_overall_best_exec_time() const {
	return job_state.get_overall_best_exec_time();
}

int64_t FrameJob::get_overall_worst_exec_time() const {
	return job_state.get_overall_worst_exec_time();
}

double FrameJob::get_overall_avg_exec_time() const {
	return job_state.get_overall_avg_exec_time();
}

void FrameJob::print_exec_stats() {
	job_state.print_exec_stats();
}


/* Implement the C-exposed interface for FrameJob objects */
#ifdef __cplusplus
extern "C" {
#endif 
void *CreateFrameJobObj(const char *fj_name, bool shareable_flag) {
	return reinterpret_cast<void *>(new FrameJob(fj_name, shareable_flag));
}
void DestroyFrameJobObj(void *fj_obj) {
	delete reinterpret_cast<FrameJob *>(fj_obj);
}
int64_t FrameJob_get_expected_exec_time_for_tid(void *fj_obj, pid_t tid) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->get_expected_exec_time_for_tid(tid);
}
uint64_t FrameJob_get_expected_required_mem_for_tid(void *fj_obj, pid_t tid) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->get_expected_required_mem_for_tid(tid);
}
int64_t FrameJob_get_wc_remaining_exec_time(void *fj_obj) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->get_wc_remaining_exec_time();
}
void FrameJob_update_wc_exec_time(void *fj_obj) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->update_wc_exec_time();
}
void FrameJob_set_wc_remaining_exec_time(void *fj_obj, int64_t ret) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->set_wc_remaining_exec_time(ret);
}
int FrameJob_prepare_job(void *fj_obj, pid_t tid, int64_t slacktime, bool first_flag) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->prepare_job(tid, slacktime, first_flag);
}
int FrameJob_release_job(void *fj_obj, pid_t tid) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->release_job(tid);
}
int64_t FrameJob_get_worst_last_exec_time(void *fj_obj) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->get_worst_last_exec_time();
}

int64_t FrameJob_get_overall_best_exec_time(void *fj_obj) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->get_overall_best_exec_time();
}

int64_t FrameJob_get_overall_worst_exec_time(void *fj_obj) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->get_overall_worst_exec_time();
}

double FrameJob_get_overall_avg_exec_time(void *fj_obj) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	return fj->get_overall_avg_exec_time();
}

void FrameJob_print_exec_stats(void *fj_obj) {
	FrameJob *fj = reinterpret_cast<FrameJob *>(fj_obj);
	fj->print_exec_stats();
}
#ifdef __cplusplus
}
#endif 

/* 
 * Object to contain run-time execution stats collection for registered frame-jobs
 * Facilitates computation of expected slack-time for frame-jobs, which are
 * sent to server to assist in job prioritization.
 * Object interface is thread-safe.
 */
FrameController::FrameController(const char *_frame_task_name, float _desired_fps,
		bool _allow_frame_drop) \
	: task_name(_frame_task_name), desired_fps(_desired_fps), allow_frame_drop(_allow_frame_drop),
	frame_start_us(0L), frame_deadline_us(0L), 
	last_drn_us(-1L), last_cpu_exec_time (-1L), max_cpu_exec_time(-1L), min_cpu_exec_time(-1L),
	avg_cpu_exec_time(-1.0),
	best_frame_us(-1L), worst_frame_us(-1L), avg_frame_us(-1.0),
	runcount(0) {
	if (desired_fps <= 0.0) {
		throw std::invalid_argument("Bad desired FPS, must be non-negative!");
	}

	// Compute desired frame duration
	int64_t micros_pf = (int64_t)round((MICROS_PER_SECOND) * 1.0 / desired_fps);
	desired_frame_drn_us = std::chrono::microseconds(micros_pf);
}
FrameController::~FrameController() {
	frame_jobs.clear(); // Destroy all FrameJobs while emptying frame_jobs
}

/* 
 * Registering a frame job returns a valid job id, or -1 on error.
 * Shareable flag parameter determines whether job is designated to be able
 * to share GPU with other processes' jobs
 */
int FrameController::register_frame_job(const char *frame_job_name, bool shareable) {
	// Use unique_lock to manage ctrl_lock
	std::unique_lock<std::mutex> lck;
	lck = std::unique_lock<std::mutex>(ctrl_lock); // Locks on construction

	frame_jobs.push_back(std::make_shared<FrameJob>(frame_job_name, shareable));
	return frame_jobs.size() - 1;
	// Unlocks on destruction of lck
}
int FrameController::unregister_frame_job(int job_id) {
	if (job_id < 0 || job_id > (long int)frame_jobs.size()-1) {
		return -1;
	}
	// Use unique_lock to manage ctrl_lock
	std::unique_lock<std::mutex> lck;
	lck = std::unique_lock<std::mutex>(ctrl_lock); // Locks on construction

	frame_jobs.erase(frame_jobs.begin() + job_id);
	return 0;
	// Unlocks on destruction of lck
}

// frame_*() functions assumed to be called by ONE thread
// Indicate the beginning and end of a frame
void FrameController::frame_start() {
	// Get current time in microseconds
	std::chrono::system_clock::time_point now_tp = std::chrono::high_resolution_clock::now();
	std::chrono::microseconds us_now = std::chrono::duration_cast<std::chrono::microseconds>(now_tp.time_since_epoch());

	frame_start_us = us_now.count();	
	frame_deadline_us = (us_now + desired_frame_drn_us).count();
}
void FrameController::frame_end() {
	// Get current time in microseconds
	std::chrono::system_clock::time_point now_tp = std::chrono::high_resolution_clock::now();
	std::chrono::microseconds us_now = std::chrono::duration_cast<std::chrono::microseconds>(now_tp.time_since_epoch());
	std::chrono::microseconds frame_drn_us(us_now.count() - frame_start_us);

	int64_t frame_us = frame_drn_us.count();
	double real_fps = 1.0 / (double)(frame_us / MICROS_PER_SECOND);
	fprintf(stderr, "Frame frame duration: %ld us, frame FPS %lf\n",
			frame_us, real_fps);

	// Update bookkeeping of frame times
	last_drn_us = frame_us;
	if (best_frame_us == -1 || frame_us < best_frame_us) { 
		best_frame_us = frame_us;
   	} 
   	if (worst_frame_us == -1 || frame_us > worst_frame_us) {
	   	worst_frame_us = frame_us;
	}
	avg_frame_us = (avg_frame_us * runcount + (double)frame_us) / (runcount+1);

	// Update estimates of CPU exec time for this frame
	// Loop over all FrameJobs to update () for each
	int64_t tot_gpu_exec_time = 0;
	for (int i=0; i< (long int)frame_jobs.size(); i++) {
		FrameJob *fji = frame_jobs[i].get();
		tot_gpu_exec_time += fji->get_worst_last_exec_time();
	}
	last_cpu_exec_time = last_drn_us - tot_gpu_exec_time;
	if (max_cpu_exec_time == -1 || max_cpu_exec_time < last_cpu_exec_time) {
		max_cpu_exec_time = last_cpu_exec_time;
	}
	if (min_cpu_exec_time == -1 || min_cpu_exec_time > last_cpu_exec_time) {
		min_cpu_exec_time = last_cpu_exec_time;
	}
	avg_cpu_exec_time = (avg_cpu_exec_time * runcount + (double)last_cpu_exec_time) / (runcount+1);

	// Every sampling period, sample the wc execution metrics
	if (runcount % SAMPLE_WC_PERIOD == 0) {
		// Use unique_lock to manage ctrl_lock
		std::unique_lock<std::mutex> lck;
		lck = std::unique_lock<std::mutex>(ctrl_lock); // Locks on construction

		// Loop over all FrameJobs to update () for each
		for (int i=0; i< (long int)frame_jobs.size(); i++) {
			FrameJob *fji = frame_jobs[i].get();
			fji->update_wc_exec_time();
		}
		// Looping from end of frame-jobs, update remaining exec time for each
		int64_t cum_wc_remaining_exec_time = 0;
		for (int j=frame_jobs.size()-1; j>=0; j--) {
			FrameJob *fjj = frame_jobs[j].get();
			fjj->wc_remaining_exec_time = cum_wc_remaining_exec_time;
			cum_wc_remaining_exec_time += fjj->wc_exec_time;
		}
	}
	runcount++;
}
int FrameController::prepare_job_by_id(int job_id, pid_t tid) {
	if (job_id < 0 || job_id >= (long int)frame_jobs.size()) {
		return -1;
	}
	// Retrieve frame_job
	std::shared_ptr<FrameJob> fj = frame_jobs[job_id];
	int64_t slacktime = 0;
	bool first_flag = true;
	/*
	 * Compute slacktime relative to absolute deadline,
	 * return -3 (dropping frame error) if allowed to skip frame and
	 * slacktime is negative 
	 */
	compute_frame_job_slacktime(tid, job_id, &slacktime, &first_flag);
	if (allow_frame_drop && slacktime < 0) {
		return -3;
	}
	// Prepare job with slacktime
	int res = fj->prepare_job(tid, slacktime, first_flag);
	if (res < 0) {
		return -2;
	}
	return 0;
}
int FrameController::release_job_by_id(int job_id, pid_t tid) {
	if (job_id < 0 || job_id >= (long int)frame_jobs.size()) {
		return -1;
	}
	// Retrieve frame_job
	std::shared_ptr<FrameJob> fj = frame_jobs[job_id];
	return fj->release_job(tid);
}
void FrameController::print_exec_stats() {
	// Loop over all FrameJobs to update () for each
	std::cout << "EXEC_STATS: FrameController (" << task_name << ") exec stats over " << runcount << " frames:" << std::endl;

	// Aggregate all the frame-units' execution stats (i.e. gpu execution stats)
	int64_t tot_min_exec_time = 0, tot_max_exec_time = 0;
	double tot_avg_exec_time = 0.0;
	for (int i=0; i< (long int)frame_jobs.size(); i++) {
		FrameJob *fji = frame_jobs[i].get();
		tot_min_exec_time += fji->get_overall_best_exec_time();
		tot_max_exec_time += fji->get_overall_worst_exec_time();
		tot_avg_exec_time += fji->get_overall_avg_exec_time();

		std::cout << "i=" << i << "\tTagged job=\"" << fji->job_name\
			<< "\", overall (min, avg, max):\t"\
			<< fji->get_overall_best_exec_time() << "us, "\
		   	<< fji->get_overall_avg_exec_time() << "us, "\
			<< fji->get_overall_worst_exec_time() << "us" << std::endl;
	}
	// Compute host execution time from total frame duration and gpu execution
	std::cout << "Execution time only on CPU (min, avg, max): \t\t\t"\
		<< min_cpu_exec_time << "us, " << avg_cpu_exec_time << "us, " \
		<< max_cpu_exec_time << "us" << std::endl;

	std::cout << "Total frame execution time (last, min, avg, max): \t\t"\
		<< last_drn_us << "us, " << best_frame_us << "us, " << avg_frame_us << "us, "\
		<< worst_frame_us << "us" << std::endl;
}

/*
* Slack-time computation at runtime relative to current absolute deadline
* which is only valid between frame_start/end() calls
* Sets input pointers on success (returning 0) with slacktime and 
* whether slacktime is initialized or not yet known
* Returns -1 on error, undefined values for input pointers
*/
int FrameController::compute_frame_job_slacktime(pid_t tid, int job_id, 
						int64_t *slacktime_p, bool *no_slacktime_p) {
	if (job_id < 0 || job_id >= (long int)frame_jobs.size()) {
		return -1;
	}
	// Use unique_lock to manage ctrl_lock
	std::unique_lock<std::mutex> lck;
	lck = std::unique_lock<std::mutex>(ctrl_lock); // Locks on construction

	// Get expected execution time for frame-job
	std::shared_ptr<FrameJob> running_fj = frame_jobs[job_id];
	int64_t exp_exec_time_us = running_fj->get_expected_exec_time_for_tid(tid);
	int64_t wc_remaining_exec_time_us = running_fj->wc_remaining_exec_time;
	if (exp_exec_time_us < 0 || wc_remaining_exec_time_us < 0) {
		*no_slacktime_p = true;
		return -1;
	}

	// Get current time in microseconds
	std::chrono::system_clock::time_point now_tp = std::chrono::high_resolution_clock::now();
	std::chrono::microseconds now_us = std::chrono::duration_cast<std::chrono::microseconds>(now_tp.time_since_epoch());
	int64_t t_us = now_us.count();
	// Compute slacktime relative to frame_deadline and remaining frame exec time
	*slacktime_p = (frame_deadline_us - t_us) - \
					   (exp_exec_time_us + wc_remaining_exec_time_us);
	*no_slacktime_p = false;
	return 0;
}


/* Implement the C-exposed interface for FrameController objects */
#ifdef __cplusplus
extern "C" {
#endif 
void *CreateFrameControllerObj(const char *frame_task_name, float desired_fps,
		bool allow_frame_drop) {
	return reinterpret_cast<void *>(
				new FrameController(frame_task_name, desired_fps, allow_frame_drop));
}
void DestroyFrameControllerObj(void *fc_obj) {
	delete reinterpret_cast<FrameController *>(fc_obj);
}
int FrameController_register_frame_job(void *fc_obj, const char *fj_name, bool shareable) {
	FrameController *fc = reinterpret_cast<FrameController *>(fc_obj);
	return fc->register_frame_job(fj_name, shareable);
}
int FrameController_unregister_frame_job(void *fc_obj, int job_id) {
	FrameController *fc = reinterpret_cast<FrameController *>(fc_obj);
	return fc->unregister_frame_job(job_id);
}
void FrameController_frame_start(void *fc_obj) {
	FrameController *fc = reinterpret_cast<FrameController *>(fc_obj);
	return fc->frame_start();
}
void FrameController_frame_end(void *fc_obj) {
	FrameController *fc = reinterpret_cast<FrameController *>(fc_obj);
	return fc->frame_end();
}
int FrameController_prepare_job_by_id(void *fc_obj, int job_id, pid_t tid) {
	FrameController *fc = reinterpret_cast<FrameController *>(fc_obj);
	return fc->prepare_job_by_id(job_id, tid);
}
int FrameController_release_job_by_id(void *fc_obj, int job_id, pid_t tid) {
	FrameController *fc = reinterpret_cast<FrameController *>(fc_obj);
	return fc->release_job_by_id(job_id, tid);
}
void FrameController_print_exec_stats(void *fc_obj) {
	FrameController *fc = reinterpret_cast<FrameController *>(fc_obj);
	return fc->print_exec_stats();
}
#ifdef __cplusplus
}
#endif 

