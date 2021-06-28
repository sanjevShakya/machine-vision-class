
# Fine tuning of a FCN-ResNet-19 model to a "floor" dataset

# Based on the PyTorch segmentation model's train.py plus the PyTorch fine tuning tutorial
# (https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

import torch
import torchvision
import PIL
import transforms as tr
import os
import utils
import time
import datetime

# Training parameters

num_epochs = 300
resume = False
print_freq = 100
num_ft_classes = 2

# Weights on the loss for each category. If obstacle pixels are rare in the training data set, give them a higher weight

floor_loss_weight = 1.0
obstacle_loss_weight = 100.0

# Custom dataset class. Modeled after torchvision.datasets.voc.VOCSegmentation

class FloorSegmentationDataset(torchvision.datasets.VisionDataset):
    """Robot Floor Segmentation Dataset.

    Args:
        root (string): Root directory of the dataset.
        image_set (string, optional): Select the image_set to use, ``train`` or ``val``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(FloorSegmentationDataset, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        image_dir = os.path.join(root, 'FloorData/Images')
        mask_dir = os.path.join(root, 'FloorData/Masks')

        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(root, 'FloorData/ImageSets')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered! Please use image_set="train" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = PIL.Image.open(self.images[index]).convert('RGB')
        target = PIL.Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_dataset(path, image_set, transform):
    ds = FloorSegmentationDataset(path, image_set=image_set, transforms=transform)
    return ds


def get_transforms(train):
    transforms = []
    base_size = 352
    crop_size = 320
    min_size = base_size
    max_size = int(2 * base_size)
    transforms.append(tr.RandomResize(min_size, max_size))
    # In training mode, perform random flips and crops
    if train:
        transforms.append(tr.RandomHorizontalFlip(0.5))
        transforms.append(tr.RandomCrop(crop_size))
    transforms.append(tr.ToTensor())
    transforms.append(tr.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return tr.Compose(transforms)


# Function to turn gradient calculations on/off depending on whether we're feature
# extracting (holding the network constant while only training the output layer)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Create instance of model class

model = torchvision.models.segmentation.fcn_resnet18(num_classes=21)

# Load pretrained weights

checkpoint = torch.load('fcn_resnet18_voc_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])

# Turn off training for pre-existing layers

#set_parameter_requires_grad(model, feature_extracting=True)

# FCN has a backbone (ResNet18) and head (attribute 'classifier' type FCNHead with 5 layers: conv, bn, relu, dropout, conv)
# model.classifier[4] is the final 21-convolution output

print('Replacing model.classifier[4] with "fresh" 2-class layer')
model.classifier[4] = torch.nn.Conv2d(128, num_ft_classes, kernel_size=(1, 1), stride=(1, 1))
device = torch.device('cuda')
model.to(device)

# Load data sets

train_dataset = get_dataset('.', "train", get_transforms(train=True))
val_dataset = get_dataset('.', "val", get_transforms(train=False))
train_sampler = torch.utils.data.RandomSampler(train_dataset)
val_sampler = torch.utils.data.SequentialSampler(val_dataset)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4,
    sampler=train_sampler, num_workers=16,
    collate_fn=utils.collate_fn, drop_last=True)
val_data_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1,
    sampler=val_sampler, num_workers=16,
    collate_fn=utils.collate_fn)

# Create optimizer

params_to_optimize = [
    {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
    {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
]
optimizer = torch.optim.SGD(
    params_to_optimize,
    lr=0.0001, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lambda x: (1 - x / (len(train_data_loader) * num_epochs)) ** 0.9)

# Define loss function

def criterion(inputs, target, weights):
    losses = {}
    for name, x in inputs.items():
        losses[name] = torch.nn.functional.cross_entropy(x, target, weight=weights, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

# Train for one epoch over the dataset

def train_one_epoch(model, criterion, class_weights, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target, class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


# Evaluate model according to IoU (intersection over union)

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


# if resume:
    # checkpoint = torch.load(args.resume, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # model_without_ddp.load_state_dict(checkpoint['model_without_ddp'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # print('Getting initial validation performance:')
    # confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
    # print(confmat)

# training loop
start_time = time.time()
best_IoU = 0.0
class_weights = torch.Tensor([floor_loss_weight, obstacle_loss_weight]).to(device)

for epoch in range(num_epochs):

    # Train one epoch

    train_one_epoch(model, criterion, class_weights, optimizer, train_data_loader, lr_scheduler, device, epoch, print_freq)

    # Test on the val dataset

    confmat = evaluate(model, val_data_loader, device=device, num_classes=num_ft_classes)
    print(confmat)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

