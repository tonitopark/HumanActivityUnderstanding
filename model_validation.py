import os
import json

import torch
from torch.autograd import Variable

from models.i3d import I3D
from dataset.videodataset import *

from argument_parser import parse_arguments

import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    num_correct_predictions = correct.sum().item()

    return num_correct_predictions / batch_size


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
            args.num_frames_test = 250

        if args.model_name == 's3d':
            args.crop_size = 224
            args.crop_method_test = 'center'
            args.crop_method_train = 'random'

    print(args)

    torch.manual_seed(args.manual_seed)

    model = I3D(num_classes=args.num_class)
    model.load_state_dict(torch.load('models/model_rgb.pth'))
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=(0,1,2,3),output_device=None)
    model.eval()

    frame_mapper = ComposeMappings([
       # ResizeFrame(256),
        CropFramePart(args.crop_size, args.crop_method_test),
        ToTensor(),
        NormalizeFrameToUnity(-1.12, 1.12)
    ])

    test_dataset = VideoDataset(
        dataset_name='kinetics',
        video_path=args.video_path,
        annotation_path=args.annotation_path,
        subset=args.test_source,
        num_clips_per_video=1,
        sample_duration=args.num_frames_test,
        frame_mapper=frame_mapper)

    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_threads,
        pin_memory=True)

    cross_entropy = torch.nn.CrossEntropyLoss()
    losses = AverageMeter()
    accuracies = AverageMeter()

    result_buffer = {'results': {}}
    # sample = np.load('v_CricketShot_g04_c01_rgb.npy').transpose(0, 4, 1, 2, 3)

    for i, (img_tensors, targets) in enumerate(test_dataset_loader):
        with torch.no_grad():
            # img_tensors = torch.autograd.Variable(torch.from_numpy(sample).cuda())

            img_tensors = Variable(img_tensors)
            _, out_logit = model(img_tensors)
            outputs = out_logit.data.cpu()
            del out_logit
            targets = targets['label']
            # targets = torch.tensor([227])

            # for img in img_tensors.permute(0,2,3,4,1).contiguous().view(-1,224,224,3):
            #     plt.imshow(img)
            #     plt.show()

            loss = cross_entropy(outputs, targets)
            accuracy = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), img_tensors.size(0))
            accuracies.update(accuracy, img_tensors.size(0))

            del loss, accuracy

            print('# ',i, '  Loss : ', losses.avg, ' Accuracy : ', accuracies.avg)

    print('Average_loss :', losses.avg)
    print('Average_accuracy : ', accuracies.avg)
