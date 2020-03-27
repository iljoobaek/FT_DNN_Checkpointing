from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import torch
import cv2
import sys
import os
import flops_counter
import time
import argparse

from vision.nn_profile import Profiler

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = '/home/droid/Downloads/kitti_data'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{:010d}.png'.format(i)) for i in range(0, 154) ]
PATH_TO_TEST_IMAGES_DIR = '/home/jetson/Workbench/CARRS_FT_int/cuMiddleware/benchmark/data/kitti_data'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{:06d}.png'.format(i)) for i in range(0, 354) ]

parse = argparse.ArgumentParser("Run an SSD with or without tagging")
parse.add_argument("--net", dest="net", default='mb1-ssd', choices=['mb1-ssd', 'vgg16-ssd'])
parse.add_argument("--path", dest="model_path", default='models/mobilenet-v1-ssd-mp-0_675.pth')
parse.add_argument("--label", dest="label_path", default='models/voc-model-labels.txt')
parse.add_argument("--fps", dest="fps", default=-1, help="Enable fps control with tagging at desired fps [default -1, disabled]")
args = parse.parse_args()

net_type = args.net
model_path = args.model_path
label_path = args.label_path
fps = args.fps
tagging_enabled = False
if str(fps) == "-1":
    print("Tagging is NOT enabled")
    time.sleep(1)
else:
    fps = float(fps)
    if fps < 0.0:
        print("Please provide a positive FPS, quitting")
        sys.exit(0)
    else:
        print("Tagging enabled, desired FPS is " + str(fps))
        tagging_enabled = True

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

# Get flops count for this network:
input_res = (1242, 375)
#import pdb; pdb.set_trace()
flops, params = flops_counter.get_model_complexity_info(predictor.net.base_net.cuda(), input_res, as_strings=True,
                                          print_per_layer_stat=False)

#######
# To enable pytorch tagging of SSD Pytorch module
if tagging_enabled:
    #sys.path.append("/home/droid/mhwork/cuMiddleware_v1/SourceCode/cu_wrapper/python")
    sys.path.append("/home/jetson/Workbench/CARRS_FT_int/cuMiddleware/python")
    import tag_layer_fps
    net = predictor.net
    allow_frame_drop = False
    fc = tag_layer_fps.FrameController("Pytorch SSD", fps, allow_frame_drop)
    tag_layer_fps.tag_pt_module_layers_at_depth(net, fc, True, 0)
######

# Set up layer-level profiling
enable_prof = False
prof = Profiler(predictor.net.base_net, enabled=enable_prof)

timer = Timer()
tot_exe = 0.
max_exe = 0.
tot_fps = 0.
min_fps =100
frm_cnt = 0
tenth_flag = 10
autograd_prof_flag = False

for image_path in TEST_IMAGE_PATHS:
    timer.start()
    tenth_flag -= 1
    orig_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_copy = orig_image.copy()
    if tenth_flag == 0 and enable_prof:
        if autograd_prof_flag:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                predictor.predict(image_copy, 10, 0.4, ret_fps=True)
        else:
            prof.set_enable(True)
            predictor.predict(image_copy, 10, 0.4, ret_fps=True)
            prof.set_enable(False)
            print("Profiling results:")
            #print(prof)

    boxes, labels, probs, fps = (None, None, None, None)
    if tagging_enabled:
        # Encapsulate the GPU work with frame-rate control
        with tag_layer_fps.FrameController.frame_context(fc) as frame:
            boxes, labels, probs, fps = predictor.predict(image_copy, 10, 0.4, ret_fps=True)
    else:
        boxes, labels, probs, fps = predictor.predict(image_copy, 10, 0.4, ret_fps=True)
    # Keep track of average FPS
    interval = timer.end()
    frame_fps = 1.0 / interval;
    frm_cnt+=1
    if frm_cnt > 15:  # ignore first a few frame result
        tot_fps += frame_fps
        tot_exe += interval
        if frame_fps < min_fps:
            min_fps = frame_fps
        if interval > max_exe:
            max_exe = interval
        print('FPS: {:.2f}'.format(frame_fps))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('SSD (PyTorch) annotated', orig_image)

    #if frm_cnt == 2: # to sync and run with other applications
    #    print("PyTorch demo is ready to go and please create run_benchmark.txt file");
    #    while True :
    #        if os.path.exists('../run_benchmark.txt') :
    #            break;
    #        cv2.waitKey(1);

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print summary of execution stats
if tagging_enabled:
    fc.print_exec_stats()

cv2.destroyAllWindows()
print('-'*60+'\n')
print('Avg : {:.2f} ms {:.2f} fps'.format( tot_exe*1000/ (frm_cnt-15) ,(tot_fps / (frm_cnt-15))))
print('Worst : {:.2f} ms {:.2f} fps'.format( max_exe*1000 ,min_fps))
if tenth_flag < 0 and enable_prof:
    print("Profiling results:")
    print(prof)

#open("../stop_benchmark.txt","w+")
