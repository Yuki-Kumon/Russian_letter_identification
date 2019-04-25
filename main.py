# -*- coding: utf-8 -*-

"""
ロシア語の手書き文字をニューラルネットで学習する練習。
Author :
    Yuki Kumon
Last Update :
    2019-04-25
"""


from torch.utils.data import Dataset
import os
import sys
import pandas as pd

LABEL_IDX = 1
IMG_IDX = 2


class MyDataset(Dataset):

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
        image = io.imread(img_name)
        # 画像へ処理を加える
        if self.transform:
            image = self.transform(image)

        return image, label
