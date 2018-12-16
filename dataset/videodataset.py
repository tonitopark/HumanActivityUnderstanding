import copy
import json
import math
import os
import fnmatch

import torch.utils.data

from dataset.frame_functions import SelectFrames
from dataset.mappings import *


def video_loader(video_path, frame_indices):
    video = []
    for i in frame_indices:
        #image_path = os.path.join(video_path, 'image_{:05d}.jpg'.format(i))
        image_path = os.path.join(video_path, 'frame{}.jpg'.format(i))
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                with Image.open(f) as img:
                    video.append(img.convert('RGB'))
        else:
            return video
    return video


def make_dataset(root_path,
                 annotation_path,
                 subset,
                 num_clips_per_video,
                 sample_duration,
                 dataset_name):
    # Load annotation data
    with open(annotation_path, 'r') as data_file:
        data = json.load(data_file)

    # get video names and annotations
    video_names = []
    annotations = []

    if dataset_name == 'activitynet':

        for key, value in data['database'].items():
            this_subset = value['subset']
            if this_subset == subset:
                if subset == 'testing':
                    video_names.append('v_{}'.format(key))
                else:
                    video_names.append('v_{}'.format(key))
                    annotations.append(value['annotations'])

        class_names = []
        index = 0
        for node1 in data['taxonomy']:
            is_leaf = True
            for node2 in data['taxonomy']:
                if node2['parentId'] == node1['nodeId']:
                    is_leaf = False
                    break
            if is_leaf:
                class_names.append(node1['nodeName'])

        # compute class to label ids
        class_to_idx = {}

        for i, class_name in enumerate(class_names):
            class_to_idx[class_name] = i


    elif dataset_name == 'kinetics':

        for key, value in data['database'].items():
            this_subset = value['subset']
            if this_subset == subset:
                if subset == 'testing' or this_subset == 'test':
                    video_names.append('test/{}'.format(key))
                    annotations.append(value['annotations'])
                elif subset == 'training' or this_subset == 'train':
                    label = value['annotations']['label']
                    video_names.append('train/{}/{}'.format(label, key))
                    annotations.append(value['annotations'])
                elif subset == 'validation' or this_subset == 'val':
                    label = value['annotations']['label']
                    label = label.replace(' ','_')
                    video_names.append('valid/{}/{}'.format(label, key))
                    annotations.append(value['annotations'])

        # compute class to label ids
        class_to_idx = {}
        index = 0
        for class_label in data['labels']:
            class_to_idx[class_label] = index
            index += 1

    # compute label to class ids
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            ('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i][:-14])
        if not os.path.exists(video_path):
            continue
        file_names = fnmatch.filter(os.listdir(video_path), '*.jpg')
        #file_names = os.listdir(video_path,)
        image_file_names = ['{:05d}'.format(int(x[5:-4])) for x in file_names if 'frame' in x]


        #image_file_names = [x for x in file_names if 'image' in x]
        image_file_names.sort(reverse=True)

        n_frames = 0
        if image_file_names:
            n_frames = int(image_file_names[0])
            #n_frames = int(image_file_names[0][6:11])

        if n_frames <= 0:
            continue

        for key, value in annotations[i].items():

            if dataset_name == 'activitynet':
                begin_t = 1  # math.ceil(annotation['segment'][0] * fps)
                end_t = n_frames  # math.ceil(annotation['segment'][1] * fps)
                video_id = video_names[i][2:]

            elif dataset_name == 'kinetics':
                begin_t = 1
                end_t = n_frames
                video_id = video_names[i][:-14].split('/')[-1]

            sample = {
                'video': video_path,
                'segment': [begin_t, end_t],
                'n_frames': n_frames,
                'video_id': video_id
            }

            if len(annotations) != 0:
                sample['label'] = class_to_idx[value]
            elif len(annotations) == 0:
                sample['label'] = -1


            if num_clips_per_video == 1:

                sample['frame_indices'] = list(range(1, n_frames + 1))
                dataset.append(sample)

            else:

                if num_clips_per_video > 1:
                    step = max(1, math.ceil((n_frames - 1 - sample_duration) /
                                         (num_clips_per_video-1)))

                for j in range(1, max(2, n_frames - sample_duration + 2 ), step):

                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(
                        range(j, min(n_frames + 1, j + max(step, sample_duration))))
                    dataset.append(sample_j)

    return dataset, idx_to_class


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_name,
                 video_path,
                 annotation_path,
                 subset,
                 num_clips_per_video='full_frames',
                 sample_duration=16,
                 frame_mapper=None,
                 get_loader = video_loader):


        self.data, self.class_names = make_dataset(
            video_path, annotation_path, subset,
            num_clips_per_video,
            sample_duration,
            dataset_name)
        self.frame_loader = get_loader
        self.frame_selector = SelectFrames(sample_duration)
        self.frame_mapper = frame_mapper

            # ComposeMappings([
            # CropFramePart(128),
            # ResizeFrame(256),
            # FlipFrame(),
            # ToTensor(),
            # NormalizeFrame([0, 0, 0], [1, 1, 1])])

    def __getitem__(self, index):
        path = self.data[index]['video']

        frame_indices = self.frame_selector(
            self.data[index]['frame_indices'],
            method='random')

        clip = self.frame_loader(path, frame_indices)

        clip = [self.frame_mapper(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]

        return clip, target

    def __len__(self):
        return len(self.data)
