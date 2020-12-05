#ifndef TAG_FRAME_H
#define TAG_FRAME_H

#include <sys/types.h> /* pid_t */
#include <stdint.h>	// int64_t
#ifdef __cplusplus
#include <deque> 	// std::deque
#include <mutex> /* std::mutex */
#include <chrono> // std::chrono::duration
#include <string> // std::string
#endif

#include "tag_state.h" /* TagState */

#ifdef __cplusplus
struct FrameJob {
	std::string job_name;
	TagState job_state; 		// Current per-thread execution stats of job
	bool shareable;				// Determines whether job can share GPU with other processes

	// (sampled) w.c. expected exec time (us) for this job for all threads using assoc.'d job_state
	// -1 if not yet initialized
	int64_t wc_exec_time;
	// (sampled) w.c. expected exec time (us) for remaining jobs in frame
	// -1 if not yet initialized
	int64_t wc_remaining_exec_time;

	FrameJob(const char* _job_name, bool shareable_flag);
	~FrameJob();
	
	/* Expected execution stats retrieval */
	int64_t get_wc_remaining_exec_time();
	int64_t get_expected_exec_time_for_tid(pid_t tid);
	uint64_t get_expected_required_mem_for_tid(pid_t tid);
	int64_t get_overall_best_exec_time() const;
	int64_t get_overall_worst_exec_time() const;
	double get_overall_avg_exec_time() const;
	int64_t get_worst_last_exec_time() const;

	/* Sample wc_exec_time */
	// This is run every N
	void update_wc_exec_time();
	void set_wc_remaining_exec_time(int64_t rem_exec_time);

	/* Job preparation/release */
	// Returns -1 if job must be aborted, 0 on successful preparation
	// On success, starts timer for TagState object
	int prepare_job(pid_t tid, int64_t slacktime, bool first_flag);
	int release_job(pid_t tid);

	void print_exec_stats();
};
#endif

/* C-exposed interface for FrameJob */
#ifdef __cplusplus
extern "C" {
#endif 
void *CreateFrameJobObj(const char *fj_name, bool shareable_flag); 
void DestroyFrameJobObj(void *fj_obj);
int64_t FrameJob_get_expected_exec_time_for_tid(void *fj_obj, pid_t tid);
uint64_t FrameJob_get_expected_required_mem_for_tid(void *fj_obj, pid_t tid);
int64_t FrameJob_get_wc_remaining_exec_time(void *fj_obj);
void FrameJob_update_wc_exec_time(void *fj_obj);
void FrameJob_set_wc_remaining_exec_time(void *fj_obj, int64_t rem_exec_time);
int FrameJob_prepare_job(void *fj_obj, pid_t tid, int64_t slacktime, bool first_flag);
int FrameJob_release_job(void *fj_obj, pid_t tid);
int64_t FrameJob_get_worst_last_exec_time(void *fj_obj);
int64_t FrameJob_get_overall_best_exec_time(void *fj_obj);
int64_t FrameJob_get_overall_worst_exec_time(void *fj_obj);
double FrameJob_get_overall_avg_exec_time(void *fj_obj);
void FrameJob_print_exec_stats(void *fj_obj);
#ifdef __cplusplus
}
#endif


/* 
 * Object to contain run-time execution stats collection for registered frame-jobs
 * Facilitates computation of expected slack-time for frame-jobs, which are
 * sent to server to assist in job prioritization.
 * Object interface is thread-safe.
 */
#define SAMPLE_WC_PERIOD 10 // Sample worst-case metrics every 10 frames
#define MICROS_PER_SECOND 1e6

#ifdef __cplusplus
struct FrameController {
	std::deque<std::shared_ptr<FrameJob> > frame_jobs;
	std::string task_name;
	float desired_fps;
	bool allow_frame_drop;
	std::chrono::microseconds desired_frame_drn_us;
	long int frame_start_us;			// Most recent frame start time (us)
	long int frame_deadline_us;			// Most recent frame deadline (us)
	int64_t last_drn_us;				// Most recent frame duration (us), -1 if uninit
	int64_t last_cpu_exec_time;			// Most recent frame cpu exec time (us), -1 if uninit
	int64_t max_cpu_exec_time;			// Max frame cpu exec time (us), -1 if uninit
	int64_t min_cpu_exec_time;			// Min frame cpu exec time (us), -1 if uninit
	double avg_cpu_exec_time;			// Avg frame cpu exec time (us), -1 if uninit

	int64_t best_frame_us;				// Min frame duration (us), -1 if uninit
	int64_t worst_frame_us;				// Max frame duration (us), -1 if uninit
	double avg_frame_us;				// Avg frame duration (us), -1 if uninit
	unsigned int runcount;
	std::mutex ctrl_lock;

	/* 
	 * Constructor:
	 * Designates FrameController name, target fps, and flag for whether to throw DroppedFrameException
	 * when slacktimes become negative.
	 */
	FrameController(const char *frame_task_name, float desired_fps, bool allow_frame_drop);
	~FrameController();

	/* Registration calls to characterize work within a "frame" of repetitive work*/
	// Jobs are designated by job_id (int)
	int register_frame_job(const char *frame_job_name, bool shareable); // -1 on error, valid job id on success
	int unregister_frame_job(int job_id); // -1 on error, 0 on success

	// Indicate the beginning and end of a frame
	void frame_start();
	void frame_end();

	/* Job preparation and release by job_id */
	// returns 0 on success, -1 bad job_id, -2 if job must abort, -3 if dropping frame
	// Note (TODO): not threadsafe! Assumes registration/unregistration will never overlap execution
	int prepare_job_by_id(int job_id, pid_t tid);
	// Returns 0 on successfully releasing GPU for job
	int release_job_by_id(int job_id, pid_t tid);

	void print_exec_stats();
	private:
		/*
		 * Slack-time computation at runtime relative to current absolute deadline (in us)
		 * which is only valid between frame_start/end() calls
		 * Sets input pointers on success (returning 0) with slacktime and 
		 * whether slacktime is initialized or not yet known
		 * Returns -1 on error, undefined values for input pointers
		 */
		int compute_frame_job_slacktime(pid_t tid, int job_id,
				int64_t *slacktime_p, bool *initialize_flag_p);
};
#endif

/* Implement the C-exposed interface for FrameController objects */
#ifdef __cplusplus
extern "C" {
#endif 
void *CreateFrameControllerObj(const char *frame_task_name, float desired_fps, bool allow_frame_drop);
void DestroyFrameControllerObj(void *fc_obj);
int FrameController_register_frame_job(void *fc_obj, const char *fj_name, bool shareable);
int FrameController_unregister_frame_job(void *fc_obj, int job_id);
void FrameController_frame_start(void *fc_obj);
void FrameController_frame_end(void *fc_obj);
int FrameController_prepare_job_by_id(void *fc_obj, int job_id, pid_t tid);
int FrameController_release_job_by_id(void *fc_obj, int job_id, pid_t tid);
void FrameController_print_exec_stats(void *fc_obj);
#ifdef __cplusplus
}
#endif

#endif /* TAG_FRAME_H */
