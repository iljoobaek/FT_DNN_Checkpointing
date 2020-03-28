import torch
import cv2
import sys
import os
import flops_counter
import time
import argparse
import pickle
import threading
import time
from resnet_library_ft import *
# Checkpoint paths to save the checkpoints to.
CHECKPOINT_1_PATH = 'cp1.pt'
CHECKPOINT_2_PATH = 'cp2.pt'
CHECKPOINT_3_PATH = 'cp3.pt'
CHECKPOINT_4_PATH = 'cp4.pt'


heartbeat_interval = 0.5
curr_frame_id = -1
frame_id_mutex = threading.Lock()
heart_beat_flag = True


parse = argparse.ArgumentParser("Run an SSD with or without tagging")
parse.add_argument("--net", dest="net", default='mb1-ssd', choices=['mb1-ssd', 'vgg16-ssd'])
parse.add_argument("--path", dest="model_path", default='models/mobilenet-v1-ssd-mp-0_675.pth')
parse.add_argument("--label", dest="label_path", default='models/voc-model-labels.txt')
parse.add_argument("--fps", dest="fps", default=-1, help="Enable fps control with tagging at desired fps [default -1, disabled]")
parse.add_argument("--ft",dest="ft",default=0)
parse.add_argument("--hbt", dest="heartbeat_input", default=heartbeat_interval)
parse.add_argument("--chkpt", dest="checkpoint_level", default=0)


def delete_checkpoints():
    if(os.path.exists(CHECKPOINT_1_PATH)):
        os.remove(CHECKPOINT_1_PATH)
    if(os.path.exists(CHECKPOINT_2_PATH)):
        os.remove(CHECKPOINT_2_PATH)
    if(os.path.exists(CHECKPOINT_3_PATH)):
        os.remove(CHECKPOINT_3_PATH)

def heartbeat_fn():
    while heart_beat_flag:
        frame_id_mutex.acquire()
        shared = {"frame_id":curr_frame_id}
        frame_id_mutex.release()
        fd = open("frame_id_processed.pkl","wb")
        pickle.dump(shared,fd)
        fd.close()
        time.sleep(heartbeat_interval)


def clean_up_files():
    try:
        os.remove("frame_id_processed.pkl")
    except:
        print("Error while deleting file ")


