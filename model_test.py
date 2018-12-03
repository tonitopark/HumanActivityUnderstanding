import os
import json

import torch
from torch.autograd import Variable

from models.i3d import I3D
from dataset.videodataset import *


from argument_parser import parse_arguments



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
        pin_memory=True)


    result_buffer = {'results':{}}

    for idx, (img_tensors, targets) in enumerate(test_dataset_loader):

        outputs,out_logit = model(img_tensors.cuda())
        outputs = outputs.data.cpu()

        video_ids = targets['video_id']

        counter=0
        prev_video_id = video_ids.pop(0)
        output_buffer = list([outputs[0].data.cpu()])
        for output, video_id in zip(outputs[1:],video_ids):

            if prev_video_id != video_id:

                avg_score = torch.mean(torch.stack(output_buffer),dim=0)
                score_sorted, class_id = torch.topk(avg_score, k=10)
                score_sorted = score_sorted.detach().numpy()
                class_id_list = class_id.numpy()

                tmp_buffer = []
                for score, class_id in zip(score_sorted, class_id_list):
                     tmp_buffer.append({
                        'class_name': str(test_dataset.class_names[class_id]),
                        'score': float(score)} )

                result_buffer['results'][video_id] = tmp_buffer
                output_buffer = []

            output_buffer.append(output.data.cpu())
            previous_video_id=video_id

        if (idx % 100 == 0):
            with open(os.path.join(args.result_path, 'test.json'), 'w') as fp:
                json.dump(result_buffer, fp)
                print(result_buffer)

    with open(
            os.path.join(args.result_path, 'test.json'), 'w') as fp:
        json.dump(result_buffer, fp)
