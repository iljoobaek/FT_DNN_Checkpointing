import os
import sys
import functools
from ctypes import cdll, c_uint, c_int, c_void_p, c_char_p, c_ulonglong, c_double, c_float, c_bool

libc = cdll.LoadLibrary('libc.so.6')
libpytag = cdll.LoadLibrary("libcarss.so")

# Modify the res and argtypes of FrameController interface
libpytag.CreateFrameControllerObj.restype = c_void_p
libpytag.CreateFrameControllerObj.argtypes= [c_char_p, c_float, c_bool]

libpytag.DestroyFrameControllerObj.restype = None
libpytag.DestroyFrameControllerObj.argtypes = [c_void_p]

libpytag.FrameController_frame_start.restype = None
libpytag.FrameController_frame_start.argtypes = [c_void_p]

libpytag.FrameController_frame_end.restype = None
libpytag.FrameController_frame_end.argtypes = [c_void_p]

libpytag.FrameController_register_frame_job.restype = c_int
libpytag.FrameController_register_frame_job.argtypes = [c_void_p, c_char_p, c_bool]

libpytag.FrameController_unregister_frame_job.restype = c_int
libpytag.FrameController_unregister_frame_job.argtypes = [c_void_p, c_int]

libpytag.FrameController_prepare_job_by_id.restype = c_int
libpytag.FrameController_prepare_job_by_id.argtypes = [c_void_p, c_int, c_int];

libpytag.FrameController_release_job_by_id.restype = c_int
libpytag.FrameController_release_job_by_id.argtypes = [c_void_p, c_int, c_int];

def gettid():
    SYS_gettid = 186 # SYS_gettid
    return libc.syscall(SYS_gettid)

class AbortJobException (Exception):
    pass

class DroppedFrameException(Exception):
    pass

class FrameController(object):
    def __init__(self, frame_name, desired_fps, allow_frame_drop):
        frame_name = c_char_p(frame_name.encode('utf8'))
        desired_fps = c_float(desired_fps)
        allow_frame_drop = c_bool(allow_frame_drop)
        self.fc = c_void_p(libpytag.CreateFrameControllerObj(frame_name, desired_fps,
                                                            allow_frame_drop))

    def __del__(self):
        libpytag.DestroyFrameControllerObj(self.fc)

    def frame_start(self):
        libpytag.FrameController_frame_start(self.fc)

    def frame_end(self):
        libpytag.FrameController_frame_end(self.fc)

    def register_frame_job(self, fj_name, shareable):
        fj_name = c_char_p(fj_name.encode('utf8'))
        shareable = c_bool(shareable)
        return libpytag.FrameController_register_frame_job(self.fc, fj_name, shareable)

    def unregister_frame_job(self, fj_id):
        fj_id = c_int(fj_id)
        return libpytag.FrameController_unregister_frame_job(self.fc, fj_id)

    def prepare_job_by_id(self, fj_id, tid):
        fj_id = c_int(fj_id)
        tid = c_int(tid)
        return libpytag.FrameController_prepare_job_by_id(self.fc, fj_id, tid)

    def release_job_by_id(self, fj_id, tid):
        fj_id = c_int(fj_id)
        tid = c_int(tid)
        return libpytag.FrameController_release_job_by_id(self.fc, fj_id, tid)

    @classmethod
    def frame_context(cls, fc):
        '''
        Nicer syntax for with clauses around already-init FC object
        example:

        fc = FrameController("Frame1", des_fps)
        ...
        for im in images:
            with FrameController.frame_context(fc) as frame:
                # Do decorated and non-wrapped frame work on im
        '''
        return fc
        
    def __enter__(self):
        # On entering context, mark the start of the frame
        self.frame_start()
        return self

    def __exit__(self, type, value, traceback):
        # On leaving context, mark the end of the frame
        self.frame_end()
        if traceback is not None:
            # Exception occurred, swallow only the DroppedFrameException
            if type == DroppedFrameException:
                print("FrameController context caught a DroppedFrameException, skipping remaining frame's work")
                return True
            else:
                return False 

def frame_job_tag_fn(fn, fc_obj, job_name, is_shareable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Using calling thread's tid, prepare wrapped FrameJob
        tid = gettid()
        prep = wrapper.fc.prepare_job_by_id(wrapper.job_id, tid)
        
        if prep < 0:
            if prep == -1:
                print("Error preparing job!")
                raise Exception("Error preparing job!")
            elif prep == -2:
                print("Aborting job, not permitted by server to run!")
                raise AbortJobException("Not permitted to run by server!")
            elif prep == -3:
                print("Skipping remaining jobs for this frame!")
                raise DroppedFrameException("Skipping remaining jobs for this frame!")

        # Do work
        res = wrapper.fn(*args, **kwargs)

        # Lastly, release job resources before returning
        wrapper.fc.release_job_by_id(wrapper.job_id, tid)
        return res

    # Initialize the wrapper function and FrameController's new FrameJob
    wrapper.fc = fc_obj
    wrapper.fn = fn
    job_id = wrapper.fc.register_frame_job(job_name, is_shareable)
    if job_id < 0:
        # Failed to register frame job!
        raise Exception("Failed to register frame job! Err %d\n" % (job_id))
    wrapper.job_id = job_id
    return wrapper

def tag_forward_at_depth(m_obj, fc, is_shareable, tgt_depth, c_depth):
    import torch
    assert(isinstance(m_obj, torch.nn.Module))
    n_children = sum(1 for _ in m_obj.children())
    if c_depth == tgt_depth or\
         (tgt_depth == -1 and n_children == 0):
        # Wrap tagging routine for forward() function at this depth
        print("Wrapping tagging routine for forward() function of module: %s" % m_obj.__class__)
        old_fwd = m_obj.forward
        new_fwd = frame_job_tag_fn(old_fwd, fc, old_fwd.__name__ + "-depth"+str(c_depth), is_shareable)
        m_obj.forward = new_fwd

    # Else, recursive case
    for child_mod in m_obj.children():
        tag_forward_at_depth(child_mod, fc, is_shareable, tgt_depth, c_depth+1)
    return

def tag_pt_module_layers_at_depth(m_obj, fc, is_shareable, tgt_depth=-1):
    '''
    Helper function to wrap tagging routine around all submodules' (at some recursive tgt_depth)
    forward() functions with frame-job object pertaining to given FrameController obj.
    Depth is indexed starting at 0. Tgt_depth of -1 wraps all children-less modules.
    '''
    import torch
    tag_forward_at_depth(m_obj, fc, is_shareable, tgt_depth, 0)


if __name__ == "__main__":
    fc = FrameController("dummy", 25.0, False)
    fc.frame_start()
    fc.frame_end()

    


