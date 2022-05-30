#!/usr/bin/env python
# coding: utf-8



# %load utils.py
import numpy as np
import os
import random
from scipy import misc
import imageio
import matplotlib.pyplot as plt
import time

os.chdir(".")




def LoadData(num_classes = 50, num_samples_per_class_train = 15, num_samples_per_class_test = 5, seed = 1):
    """
    Load data and split it into training and testing
    Args:
        num_classes: number of classes adopted, -1 represents using all the classes
        num_samples_per_class_train: number of samples per class used for training
        num_samples_per_class_test: number of samples per class used for testing
        seed: random seed to ensure consistent results
    Returns:
        a tuple of (1) images for training (2) labels for training (3) images for testing, and (4) labels for testing
            (1) numpy array of shape [num_classes * num_samples_per_class_train, 784], binary pixels
            (2) numpy array of shape [num_classes * num_samples_per_class_train], integers of the class label
            (3) numpy array of shape [num_classes * num_samples_per_class_test, 784], binary pixels
            (4) numpy array of shape [num_classes * num_samples_per_class_test], integers of the class label
    """
    random.seed(seed)
    np.random.seed(seed)
    num_samples_per_class = num_samples_per_class_train + num_samples_per_class_test
    assert num_classes <= 1623
    assert num_samples_per_class <= 20
    dim_input = 28 * 28   # 784
    
    # construct folders
    data_folder = './omniglot_resized'
    character_folders = [os.path.join(data_folder, family, character)
                         for family in os.listdir(data_folder)
                         if os.path.isdir(os.path.join(data_folder, family))
                         for character in os.listdir(os.path.join(data_folder, family))
                         if os.path.isdir(os.path.join(data_folder, family, character))]
    random.shuffle(character_folders)
    if num_classes == -1:
        num_classes = len(character_folders)
    else:
        character_folders = character_folders[: num_classes]
    
    # read images
    all_images = np.zeros(shape = (num_samples_per_class, num_classes, dim_input))
    all_labels = np.zeros(shape = (num_samples_per_class, num_classes))
    label_images = get_images(character_folders, list(range(num_classes)), nb_samples = num_samples_per_class, shuffle = True)
    temp_count = np.zeros(num_classes, dtype=int)
    for label,imagefile in label_images:
        temp_num = temp_count[label]
        all_images[temp_num, label, :] = image_file_to_array(imagefile, dim_input)
        all_labels[temp_num, label] = label
        temp_count[label] += 1
    
    # split and random permutate
    train_image = all_images[:num_samples_per_class_train].reshape(-1,dim_input)
    test_image  = all_images[num_samples_per_class_train:].reshape(-1,dim_input)
    train_label = all_labels[:num_samples_per_class_train].reshape(-1)
    test_label  = all_labels[num_samples_per_class_train:].reshape(-1)
    train_image, train_label = pair_shuffle(train_image, train_label)
    test_image, test_label = pair_shuffle(test_image, test_label)
    return train_image, train_label, test_image, test_label 
 

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler([pathstr for pathstr in os.listdir(path) if pathstr[-4:] == '.png' ])]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels

def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image

def pair_shuffle(array_a, array_b):
    """
    Takes an image array and a label array
    Returns:
        the shuffled image array and label array
    """
    temp_perm = np.random.permutation(array_a.shape[0])
    array_a = array_a[temp_perm]
    array_b = array_b[temp_perm]
    return array_a, array_b





# %load pytorch_example.py
# import libraries
import argparse
import numpy as np
import torch
# from utils import *
from torch import nn as nn
from torch import optim
from torch.autograd import Variable





# define settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=50, 
                    help='number of classes used')
parser.add_argument('--num_samples_train', type=int, default=15, 
                    help='number of samples per class used for training')
parser.add_argument('--num_samples_test', type=int, default=5, 
                    help='number of samples per class used for testing')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed')
#args = parser.parse_args()
args =parser.parse_known_args()[0]




# define you model, loss functions, hyperparameters, and optimizers
# CNN as benchmark

start_1 = time.time()
# 记录起始时间

class CNN_bm(nn.Module):
    def __init__(self):
        super(CNN_bm, self).__init__()
        #feature提取
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,stride = 1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #分类层
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 50)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0),-1)
        x = self.layer2(x)
        return nn.functional.softmax(x)

model = CNN_bm()

#loss functions: 
criterion = nn.CrossEntropyLoss()


#hyperparameters
epochs = 3000
learning_rate = 0.1
batchsize = 64

#optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# load data
train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train, args.num_samples_test, args.seed)
# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions

#train_label = train_label.reshape(-1,1)

train_image_t = torch.FloatTensor(train_image)
train_label_t = torch.LongTensor(train_label)
train_dataset = torch.utils.data.TensorDataset(train_image_t, train_label_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,shuffle=True)

test_image_t = torch.FloatTensor(test_image)
test_label_t = torch.LongTensor(test_label)
test_dataset = torch.utils.data.TensorDataset(test_image_t, test_label_t)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,shuffle=True)


loss_list = []
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for img,label in train_loader:
        out = model(img.reshape(-1, 1, 28, 28))
        loss = criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data

    if (epoch+1)%100 == 0:
        print('epoch: {}, Train Loss: {:.6f}'.format(epoch+1, train_loss/len(train_image)))

    loss_list.append(train_loss/len(train_image))



end_1 = time.time()
print("Your running time_1 is:",int(end_1-start_1),'s')
# 计算运行时间




import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.plot(loss_list)
plt.ylabel("train loss")
plt.xlabel("epoch")
plt.show()





# get predictions on test_image
model.eval()

test_loss = 0
test_acc = 0
with torch.no_grad():
    for img,label in test_loader:  
        out = model(img.reshape(-1, 1, 28, 28))
        loss = criterion(out,label)
        test_loss += loss.data
        _, y_pred = torch.max(out, 1)
        num_correct = (y_pred == label).sum()
        test_acc += num_correct.item()

# evaluation

print("Test Accuracy:", test_acc/len(test_image))
print("Test Loss:", test_loss/len(test_image))
# note that you should not use test_label elsewhere

