from __future__ import print_function, division

import os
import subprocess
import sys

import cv2 as opencv
import numpy as np


def compute_opticalflow(class_sample_path):
    frame_names = sorted([names for names in os.listdir(class_sample_path) if names.endswith('.jpg')])
    frame_paths = [os.path.join(class_sample_path, frame_name) for frame_name in frame_names]

    img = opencv.imread(frame_paths[0])
    prvs = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)

    for index in range(1, len(frame_paths)):
        next = opencv.cvtColor(opencv.imread(frame_paths[index]), opencv.COLOR_BGR2GRAY)

        flow = optical_flow.calc(prvs, next, None)

        output_file_name = frame_paths[index].replace('image', 'flow').replace('.jpg', '.npy')

        np.save(output_file_name, flow)

        prvs = next


def video_to_opticalflow_for_class(img_dir, class_name, optical_flow):
    class_path = os.path.join(img_dir, class_name)
    if not os.path.isdir(class_path):
        return

    for sample_name in os.listdir(class_path):
        class_sample_path = os.path.join(class_path, sample_name)

        subprocess.run(compute_opticalflow(class_sample_path), shell=True)


if __name__ == "__main__":
    image_dir = sys.argv[1]

    optical_flow = opencv.DualTVL1OpticalFlow_create()

    for class_name in os.listdir(image_dir):
        video_to_opticalflow_for_class(image_dir, class_name, optical_flow)

    if class_name == 'test':
        video_to_opticalflow_for_class(image_dir, class_name, optical_flow)
