from __future__ import print_function, division

import os
import subprocess
import sys


def video_to_images_for_one_class(video_dir, img_dir, class_name):
    class_path = os.path.join(video_dir, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(img_dir, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    for file_name in os.listdir(class_path):
        if '.mp4' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_class_path, name)

        video_file_path = os.path.join(class_path, file_name)

        if not os.path.exists(dst_directory_path):
            os.mkdir(dst_directory_path)

        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 -qscale:v 2 \"{}/image_%05d.jpg\"'.format(video_file_path,
                                                                                           dst_directory_path)

        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    video_dir = sys.argv[1]
    image_dir = sys.argv[2]

    for class_name in os.listdir(video_dir):
        video_to_images_for_one_class(video_dir, image_dir, class_name)

    class_name = 'test'
    video_to_images_for_one_class(video_dir, image_dir, class_name)
