# -*- coding: utf-8 -*-

"""
ロシア語の手書き文字をニューラルネットで学習する練習。
[参考]
https://qiita.com/mckeeeen/items/e255b4ac1efba88d0ca1
https://qiita.com/sheep96/items/0c2c8216d566f58882aa
[Dataset]
https://www.kaggle.com/olgabelitskaya/classification-of-handwritten-letters
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
from torch.utils.data.sampler import SubsetRandomSampler  # データセット分割
# from torch.autograd import Variable
import os
# import sys
import pandas as pd
# import cv2
from PIL import Image
import numpy as np

LABEL_IDX = 1
IMG_IDX = 2

# file path settings
cwd = os.getcwd()
# input_file_path = '/Users/yuki_kumon/Documents/python/Russian_letter_identification/data/letters2/letters2.csv'
# ROOT_DIR = '/Users/yuki_kumon/Documents/python/Russian_letter_identification/data/letters2/'
input_file_path = os.path.join(cwd, 'data/letters2/letters2.csv')
ROOT_DIR = os.path.join(cwd, 'data/letters2/')


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
        # img_name = os.path.join(self.root_dir, 'classification-of-handwritten-letters','letters2', self.image_dataframe.iat[idx, IMG_IDX])
        img_name = os.path.join(self.root_dir, self.image_dataframe.iat[idx, IMG_IDX])
        # 画像の読み込み
        # image = io.imread(img_name)
        image = Image.open(img_name)
        # 画像へ処理を加える
        if self.transform:
            image = self.transform(image)
        return image, label


"""
class MyNormalize:
    '''
    self define normalize
    '''

    def __call__(self, image):
        '''
        write my transform here
        '''
        shape = image.shape
        image = (image - torch.mean(image)) / torch.std(image) * 16 + 64
        return image
"""


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
        # →畳み込み層2→活性化関数→プーリング層2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 全結合層に渡すためにデータを1次元に変換
        x = x.view(-1, self.num_flat_features(x))
        # 全結合層1→活性化関数
        x = F.relu(self.fc1(x))
        # 全結合層2に渡す前にドロップアウト(今回は省略)
        # x = F.dropout(x, training=self.training)
        # 全結合層2→ソフトマックス関数
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def num_flat_features(self, x):
        # Conv2dは入力を4階のテンソルとして保持する(サンプル数*チャネル数*縦の長さ*横の長さ)
        # よって、特徴量の数を数える時は[1:]でスライスしたものを用いる
        size = x.size()[1:]
        # 特徴量の数=チャネル数*縦の長さ*横の長さを計算する
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


"""
def my_collate_fn(batch):
    '''
    self define collate_fn
    '''
    # datasetの出力が
    # [image, target] = dataset[batch_idx]
    # の場合.
    images = []
    targets = []
    for sample in batch:
        image, target = sample
        images.append(image)
        targets.append(targets)
    images = torch.stack(images, dim=0)
    return [images, targets]
"""


# create Dataset
imgDataset = MyDataset(input_file_path, ROOT_DIR, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    ]))

# Creating data indices for training and validation splits:
validation_split = 0.2
shuffle_dataset = True
random_seed = 42
dataset_size = len(imgDataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# create dataloader
# train_loader = torch.utils.data.DataLoader(imgDataset, batch_size=64, sampler=train_sampler, collate_fn=my_collate_fn)
# test_loader = torch.utils.data.DataLoader(imgDataset, batch_size=100, sampler=valid_sampler, collate_fn=my_collate_fn)
train_loader = torch.utils.data.DataLoader(imgDataset, batch_size=64, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(imgDataset, batch_size=100, sampler=valid_sampler)
# hoge_loader = torch.utils.data.DataLoader(imgDataset, batch_size=64, shuffle=True)

# prepare for train
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    '''
    training function
    '''
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        # Variable型への変換(統合されたので省略)
        # image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test():
    '''
    testing function
    '''
    # initialize
    test_loss = 0.0
    correct = 0.0
    model.eval()
    for (image, label) in test_loader:
        # Variable型への変換(統合されたので省略)
        # image, label = Variable(image.float(), volatile=True), Variable(label)
        output = model(image)
        test_loss += criterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# exac
for epoch in range(1, 1000 + 1):
    train(epoch)
    test()

# save
PATH = '/Users/yuki_kumon/Documents/python/Russian_letter_identification/'
torch.save(model.state_dict(), PATH)

"""
image_test = cv2.imread('./data/letters2/33_223.png')
print(image_test)
"""
