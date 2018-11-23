import torch.utils.data

from Projects.HAV.dataset.videodataset import VideoDataset

# For Kinetics
print("Intitialize Dataset")
training_data = VideoDataset('/media/tony/DATASET/K400_JPG/',
                             '/media/tony/DATASET/K400_JPG/kinetics.json',
                             'validation',
                             dataset_name='kinetics')

print("Initialize DataLoader")
train_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

print("Test the 1st batch")
for (inputs, targets) in train_loader:
    print(inputs)
    print(targets)
    break

print('done')
