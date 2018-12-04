import os
import json

import torch
from torch.autograd import Variable

from models.i3d import I3D
from dataset.videodataset import *


from argument_parser import parse_arguments
import pandas as pd


################################################################################
# Metrics
################################################################################

def compute_video_hit_at_k(ground_truth, prediction, top_k=3, avg=False):
    """Compute accuracy at k prediction between ground truth and
    predictions data frames. This code is greatly inspired by evaluation
    performed in Karpathy et al. CVPR14.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 'label']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'label', 'score']

    Outputs
    -------
    acc : float
        Top k accuracy score.
    """
    video_ids = np.unique(ground_truth['video_id'].values)
    avg_hits_per_vid = np.zeros(video_ids.size)
    for i, vid in enumerate(video_ids):
        pred_idx = prediction['video_id'] == vid
        if not pred_idx.any():
            continue
        this_pred = prediction.loc[pred_idx].reset_index(drop=True)
        # Get top K predictions sorted by decreasing score.
        sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        # Get labels and compare against ground truth.
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video_id'] == vid
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        avg_hits_per_vid[i] = np.mean([1 if this_label in pred_label else 0
                                       for this_label in gt_label])
        if not avg:
            avg_hits_per_vid[i] = np.ceil(avg_hits_per_vid[i])
    return float(avg_hits_per_vid.mean())




if __name__ == '__main__':

    args = parse_arguments()
    if args.dataset_path:
        args.video_path = os.path.join(args.dataset_path, args.video_dir_name)
        args.annotation_path = os.path.join(args.dataset_path, args.video_dir_name, args.annotation_file_name)
        args.result_path = os.path.join(args.dataset_path, args.video_dir_name, args.result_dir_name)

    if args.dataset_name:
        if args.dataset_name == 'kinetics400':
            args.num_class = 400
        elif args.dataset_name == 'kinetics600':
            args.num_class = 600
        elif args.dataset_name == 'activitynet':
            args.num_class = 200

    if args.model_name:
        if args.model_name == 'i3d':
            args.crop_size = 224
            args.crop_method_test = 'center'
            args.crop_method_train = 'random'
            args.num_frames_test = 24

        if args.model_name == 's3d':
            args.crop_size = 224
            args.crop_method_test = 'center'
            args.crop_method_train = 'random'

    print(args)

    torch.manual_seed(args.manual_seed)

    model = I3D(num_classes=args.num_class)
    model.eval()
    model.load_state_dict(torch.load('models/model_rgb.pth'))
    model.cuda()
    model = torch.nn.DataParallel(model,device_ids=None)


    frame_mapper = ComposeMappings([
        ResizeFrame(256),
        CropFramePart(args.crop_size, args.crop_method_test),
        ToTensor(),
        NormalizeFrameToUnity(-1,1)
    ])


    test_dataset = VideoDataset(
        dataset_name='kinetics',
        video_path=args.video_path,
        annotation_path=args.annotation_path,
        subset=args.test_source,
        num_clips_per_video=6,
        sample_duration=args.num_frames_test,
        frame_mapper=frame_mapper)

    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_threads,
        pin_memory=False)




    result_buffer = {'results':{}}

    for idx, (img_tensors, targets) in enumerate(test_dataset_loader):

        outputs,out_logit = model(img_tensors.cuda())
        outputs = outputs.data.cpu()

        video_ids = targets['video_id']


        given_labels = targets['label']




        prev_video_id = video_ids.pop(0)
        output_buffer = list([outputs[0].data.cpu()])
        for output, video_id , label in zip(outputs[1:],video_ids,given_labels):

            if prev_video_id != video_id:

                score_average = torch.mean(torch.stack(output_buffer),dim=0)
                score_sorted, class_id = torch.topk(score_average, k=args.num_topk)
                score_sorted = score_sorted.detach().numpy()
                class_id_list = class_id.numpy()

                prediction_buffer = []
                for rank, (score, class_id) in enumerate(zip(score_sorted, class_id_list)):
                     prediction_buffer.append({
                        'rank' : rank,
                        'class_name': str(test_dataset.class_names[class_id]),
                        'class_id': int(class_id),
                        'score': float(score)} )

                if int(label) != -1:
                    ground_truth = {'class_name': test_dataset.class_names[int(label)],
                                'class_id' : int(label)}
                else:
                    ground_truth = {'class_name': 'Not Given',
                                'class_id' : int(label)}

                result_buffer['results'][video_id] = {'prediction':  prediction_buffer, 'ground_truth': ground_truth}
                output_buffer = []

            output_buffer.append(output.data.cpu())
            previous_video_id=video_id


        # counter +=1
        # print(counter)
        # if counter > 3:
        #     break

        if (idx % 100):
            with open(
                    os.path.join(args.result_path, 'test.json'), 'w') as fp:
                json.dump(result_buffer, fp)

   # compute top k accuracy

    ground_truth_video_list, ground_truth_label_list = [], []
    prediction_video_list, prediction_label_list, prediction_score_list=[], [], []

    for video_ids,results in result_buffer['results'].items():

        ground_truth_video_list.append(video_ids)
        ground_truth_label_list.append(results['ground_truth']['class_name'])

        for pred in results['prediction']:
            prediction_video_list.append(video_ids)
            prediction_label_list.append(pred['class_name'])
            prediction_score_list.append(pred['score'])


    ground_truth = pd.DataFrame({'video_id': ground_truth_video_list,
                                 'label': ground_truth_label_list})

    ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)

    prediction = pd.DataFrame({'video_id': prediction_video_list,
                               'label': prediction_label_list,
                               'score': prediction_score_list})

    hit_at_k = compute_video_hit_at_k(ground_truth,
                                      prediction, top_k=5)

    print(hit_at_k)

    with open(
            os.path.join(args.result_path, 'test.json'), 'w') as fp:
        json.dump(result_buffer, fp)









