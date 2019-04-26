# -*- coding: utf-8 -*-

"""
ロシア語の手書き文字をニューラルネットで学習する練習。
[参考]
https://qiita.com/mckeeeen/items/e255b4ac1efba88d0ca1
https://qiita.com/sheep96/items/0c2c8216d566f58882aa
Author :
    Yuki Kumon
Last Update :
    2019-04-26
"""


from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連
import os
import sys
import pandas as pd
import cv2
import numpy as np

LABEL_IDX = 1
IMG_IDX = 2

# file path settings
input_file_path = './data/letters2/letters2.csv'
ROOT_DIR = './data/letters2/'


# define dataset
class MyDataset(Dataset):
    '''
    dataset class
    '''

    def __init__(self, csv_file_path, root_dir, transform=None):
        # pandasでcsvデータの読み出し
        self.image_dataframe = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        # 画像データへの処理
        self.transform = transform

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        # dataframeから画像へのパスとラベルを読み出す
        label = self.image_dataframe.iat[idx, LABEL_IDX]
        img_name = os.path.join(self.root_dir, 'classification-of-handwritten-letters',
                'letters2', self.image_dataframe.iat[idx, IMG_IDX])
        # 画像の読み込み
        # image = io.imread(img_name)
        image = cv2.imread(img_name)
        # 画像へ処理を加える
        if self.transform:
            image = self.transform(image)

        return image, label


class MyNormalize:
    '''
    self define normalize
    '''

    def __call__(self, image):
        '''
        write my transform here
        '''
        shape = image.shape
        image = (image - np.mean(image)) / np.std(image) * 16 + 64
        return image


# define my network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 畳み込み層(サンプル数、チャネル数、窓のサイズ)
        self.conv1 = nn.Conv2d(4, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 34)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 入力→畳み込み層1→活性化関数→プーリング層1
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# create Dataset
imgDataset = MyDataset(input_file_path, ROOT_DIR, transform=transforms.Compose([
    transforms.ToTensor(),
    MyNormalize()
    ]))


# split dataset
train_size = int(0.8 * len(imgDataset))
test_size = len(imgDataset) - train_size
train_data, test_data = torch.utils.data.random_split(imgDataset, [train_size, test_size])

# create dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

# create network
net = Net()

"""
image_test = cv2.imread('./data/letters2/33_223.png')
print(image_test)
"""
