import torch
import cv2
import sys
import os
import time
import argparse
import pickle
import threading
import time
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet_library_ft import *


# Checkpoint paths to save the checkpoints to.
CHECKPOINT_1_PATH = 'cp1.pt'
CHECKPOINT_2_PATH = 'cp2.pt'
CHECKPOINT_3_PATH = 'cp3.pt'
CHECKPOINT_4_PATH = 'cp4.pt'

device = torch.device('cuda')

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


def initialize_model(architecture):
    model = GetResnet().get_model(architecture).to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_data(data_path):
    image_transform = transforms.Compose([
                           transforms.CenterCrop((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])
    images = datasets.ImageFolder(data_path, image_transform)
    images_iterator = torch.utils.data.DataLoader(images)
    return images_iterator



def run_model(model, images_iterator):
    frame_number = 1
    for (x, y) in images_iterator:
        x = x.to(device)
        y = y.to(device)
        model(x)
        print('Frame Number : ', frame_number)
        frame_number += 1
        break

if __name__ == '__main__':
    PATH_TO_TEST_IMAGES_DIR = '/home/jetson/Workbench/CARRS_FT_int/cuMiddleware/benchmark/data/kitti_data_resnet'
    model = initialize_model('resnet18')
    data = get_data(PATH_TO_TEST_IMAGES_DIR)
    run_model(model, data)
