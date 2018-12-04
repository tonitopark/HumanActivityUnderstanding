
import os
import json
import numpy as np
import pandas as pd
from argument_parser import parse_arguments

# compute top k accuracy
# refernce Karpathy et al. CVPR14.

if __name__ == '__main__':

    avg = False

    args = parse_arguments()
    if args.dataset_path:
        args.result_path = os.path.join(args.dataset_path, args.video_dir_name, args.result_dir_name)


    with open(os.path.join(args.result_path, 'test.json'), 'r') as result_file:
        result_buffer = json.load(result_file)

    ground_truth_video_list, ground_truth_label_list = [], []
    prediction_video_list, prediction_label_list, prediction_score_list = [], [], []

    for video_ids, results in result_buffer['results'].items():

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


    unique_video_ids = np.unique(ground_truth['video_id'].values)

    top_k_hit_vector = np.zeros(unique_video_ids.size)

    for idx, video_id in enumerate(unique_video_ids):

        idx_prediction = prediction['video_id'] == video_id

        if not idx_prediction.any():
            continue

        selected_prediction = prediction.loc[idx_prediction].reset_index(drop=True)

        # Make sure it is sorted by score
        sort_idx = selected_prediction['score'].values.argsort()[::-1][:args.num_top_k]

        selected_prediction = selected_prediction.loc[sort_idx].reset_index(drop=True)

        # Get labels and compare against ground truth.
        selected_prediction_labels = selected_prediction['label'].tolist()

        idx_ground_truth = ground_truth['video_id'] == video_id

        label_ground_truth = ground_truth.loc[idx_ground_truth]['label'].tolist()

        top_k_hit_vector[idx] = np.mean([1 if this_label in selected_prediction_labels else 0
                                         for this_label in label_ground_truth])
        if not avg:
            top_k_hit_vector[idx] = np.ceil(top_k_hit_vector[idx])

    top_k_accuracy = float(top_k_hit_vector.mean())

    print('The Top-{} Accuary is '.format(args.num_top_k),top_k_accuracy)
