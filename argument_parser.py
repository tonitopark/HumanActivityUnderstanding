import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        default='kinetics400',
        type=str,
        help='Dataset options are (activitynet | kinetics400 | kinetics600 | youtube8m)'
    )
    parser.add_argument(
        '--dataset_path',
        default='/media/tony/DATASET',
        type=str,
        help='Path of the Datset'
    )
    parser.add_argument(
        '--video_dir_name',
        default='K400_JPG',
        type=str,
        help='Name of the directory where .jpg frames are located'
    )
    parser.add_argument(
        '--annotation_file_name',
        default='kinetics.json',
        type=str,
        help='name of the .json annotation file'
    )
    parser.add_argument(
        '--result_dir_name',
        default='results',
        type=str,
        help='name of the directory to store the result'
    )
    parser.add_argument(
        '--manual_seed',
        default=1,
        type=int,
        help='Set random seed for reproducibility')
    parser.add_argument(
        '--model_name',
        default='i3d',
        type=str,
        help='model name can be '
    )

    parser.add_argument(
        '--test_source',
        default='validation',
        type=str,
        help='Test can be performed on ( test | validation ) dataset'

    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Size of the batch')
    parser.add_argument(
        '--num_threads',
        default=4,
        type=int,
        help='Number of threads for computation ')

    args = parser.parse_args()

    return args