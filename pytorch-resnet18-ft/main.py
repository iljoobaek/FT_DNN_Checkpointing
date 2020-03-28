import torch
import cv2
import sys
import os
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
resnet_model_choices = GetResnet().get_model_choices()

parse = argparse.ArgumentParser("Run an SSD with or without tagging")
parse.add_argument("--net", dest="net", default='resnet18', choices=resnet_model_choices)
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


def run_model(model, frame_paths):
    frame_number = 1
    for image_path in frame_paths:
        orig_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_copy = orig_image.copy()
        model.forward(image_copy)
        print('Frame Number : ', frame_number)
        frame_number += 1

if __name__ == '__main__':
    PATH_TO_TEST_IMAGES_DIR = 'D:/Tensorflow/data/kitti_data'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{:010d}.png'.format(i)) for i in range(0, 154)]
    resnet_model = GetResnet().get_model('resnet18')
    run_model(resnet_model, TEST_IMAGE_PATHS)
