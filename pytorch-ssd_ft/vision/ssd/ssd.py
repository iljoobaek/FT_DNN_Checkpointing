import torch.nn as nn
import torch
import os
import numpy as np
from typing import List, Tuple
import time
import torch.nn.functional as F
from ..utils import box_utils
from collections import namedtuple
import pickle
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #
# Define a common checkpoint path

cwd_path = '/home/jetson/Workbench/cuMiddleware/benchmark/nandhakishore/cmu-pytorch-ssd/'
CHECKPOINT_1_PATH = cwd_path + 'cp1.pt'
CHECKPOINT_2_PATH = cwd_path + "cp2.pt"
CHECKPOINT_3_PATH = cwd_path + "cp3.pt"
#CHECKPOINT_1_PATH = 'cp1.pt'
#CHECKPOINT_2_PATH = 'cp2.pt'
#CHECKPOINT_3_PATH = 'cp3.pt'


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        end_layer_index = 0
        frame_start_time = time.time()
        if((not(os.path.exists(CHECKPOINT_2_PATH))) and (not(os.path.exists(CHECKPOINT_3_PATH)))):
            if(os.path.exists(CHECKPOINT_1_PATH)):
                x = torch.load(CHECKPOINT_1_PATH)
                fd1 = open("confidence.pkl","rb")
                confidences = pickle.load(fd1)
                fd2 = open("locations.pkl","rb")
                locations = pickle.load(fd2)
                fd3 = open("end_layer_index.pkl","wb")
                end_layer_index = pickle.load(fd3)
                print("Checkpoint1 restored..recovering...")
            else:
                for end_layer_index in self.source_layer_indexes:
                    if isinstance(end_layer_index, GraphPath):
                        path = end_layer_index
                        end_layer_index = end_layer_index.s0
                        added_layer = None
                    elif isinstance(end_layer_index, tuple):
                        added_layer = end_layer_index[1]
                        end_layer_index = end_layer_index[0]
                        path = None
                    else:
                        added_layer = None
                        path = None
                    for layer in self.base_net[start_layer_index: end_layer_index]:
                        x = layer(x)
                    if added_layer:
                        y = added_layer(x)
                    else:
                        y = x
                    if path:
                        sub = getattr(self.base_net[end_layer_index], path.name)
                        for layer in sub[:path.s1]:
                            x = layer(x)
                        y = x
                        for layer in sub[path.s1:]:
                            x = layer(x)
                        end_layer_index += 1
                    start_layer_index = end_layer_index
                    confidence, location = self.compute_header(header_index, y)
                    header_index += 1
                    confidences.append(confidence)
                    locations.append(location)
                
                frame_cp1_pickle_before = time.time()
                fd1 = open("confidence.pkl","wb")
                pickle.dump(confidences,fd1)   
                fd2 = open("locations.pkl","wb")
                pickle.dump(locations,fd2)
                fd3 = open("end_layer_index.pkl","wb")
                pickle.dump(end_layer_index,fd3)
                frame_cp1_save_before = time.time()
                try:
                    torch.save(x,CHECKPOINT_1_PATH)
                except KeyboardInterrupt:
                    torch.save(x,CHECKPOINT_1_PATH)
                print("Source layer done, checkpoint 1 saved")
            frame_cp1_after = time.time()
        if(not(os.path.exists(CHECKPOINT_3_PATH))):
            if(os.path.exists(CHECKPOINT_2_PATH)):
                x = torch.load(CHECKPOINT_2_PATH)
                print("Checkpoint2 restored..recovering...")
                
            else:
                for layer in self.base_net[end_layer_index:]:
                    x = layer(x)
                frame_cp2_save_before = time.time()
                try:
                    torch.save(x,CHECKPOINT_2_PATH)
                except KeyboardInterrupt:
                    torch.save(x,CHECKPOINT_2_PATH)
            print("Base net layer done, checkpoint 2 saved")
            if(os.path.exists(CHECKPOINT_1_PATH)):
                os.remove(CHECKPOINT_1_PATH)

        frame_cp2_after = time.time()
        if(os.path.exists(CHECKPOINT_3_PATH)):
            x = torch.load(CHECKPOINT_3_PATH)
            fd1 = open("confidence.pkl","rb")
            confidences = pickle.load(fd1)
            fd2 = open("locations.pkl","rb")
            locations = pickle.load(fd2)
            print("Checkpoint3 restored..recovering...")
            
        else:
            for layer in self.extras:
                x = layer(x)
                confidence, location = self.compute_header(header_index, x)
                header_index += 1
                confidences.append(confidence)
                locations.append(location)
            frame_cp3_pickle_before = time.time()
            fd1 = open("confidence.pkl","wb")
            pickle.dump(confidences,fd1)   
            fd2 = open("locations.pkl","wb")
            pickle.dump(locations,fd2)
            frame_cp3_save_before = time.time()
            try:
                torch.save(x,CHECKPOINT_3_PATH)
            except KeyboardInterrupt:
                torch.save(x,CHECKPOINT_3_PATH)
            print("Extra layers done, checkpoint 3 saved")
            if(os.path.exists(CHECKPOINT_2_PATH)):
                os.remove(CHECKPOINT_2_PATH)
        frame_cp3_after = time.time()
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        print('Frame start time:', 0)
        print('CP1 before pickle save: ', frame_cp1_pickle_before - frame_start_time)
        print('CP1 after pickle save: ', frame_cp1_save_before - frame_cp1_pickle_before)
        print('Cp1 after torch.save: ', frame_cp1_after - frame_cp1_save_before)
        print('CP1 overall time:', frame_cp1_after - frame_start_time)
        print('CP2 before torch.save:', frame_cp2_save_before - frame_cp1_after)
        print('CP2 after torch.save and CP3 start:', frame_cp2_after - frame_cp2_save_before)
        print('CP2 overall time:', frame_cp2_after - frame_cp1_after)
        print('CP3 before pickle save: ', frame_cp3_pickle_before - frame_cp2_after)
        print('Cp3 after pickle save: ', frame_cp3_save_before - frame_cp3_pickle_before)
        print('CP3 after torch.save:', frame_cp3_after - frame_cp3_save_before)
        print('Cp2 overall time: ', frame_cp3_after - frame_cp2_after)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations


    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
