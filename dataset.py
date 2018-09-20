from torch.utils.data import Dataset
import json
import os
import math
import random
import urllib3
import zipfile
import h5py
import numpy as np
import config


class Gaze(object):

    def __init__(self, cfg=config.default, transform=None, target_transform=None):

        self.cfg = cfg
        self.transform = transform
        self.target_transform = target_transform

        self._check_files()

        self.data = h5py.File(os.path.join(self.cfg.paths.data_folder, self.cfg.paths.h5_file), 'r')
        with open(os.path.join(self.cfg.paths.data_folder, self.cfg.paths.json_file), 'r') as json_file:
            self.table = json.loads(json_file.read())
        self.indexes = list(self.table['image_path'].keys())

        random.shuffle(self.indexes)

    def __getitem__(self, index):

        path = self.table['image_path'][self.indexes[index]]
        img = np.array(self.data['image'].get(path), dtype=np.float32)
        img = img.reshape([1, img.shape[0], img.shape[1]])
        target = np.array(eval(self.table['eye_details'][self.indexes[index]]['look_vec']), dtype=np.float32)[0:3]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):

        return len(self.table['image_path'])

    def _check_files(self):

        flag_download = False

        if not os.path.exists(self.cfg.paths.data_folder):
            os.makedirs(self.cfg.paths.data_folder)
            flag_download = True

        if not flag_download and (not os.path.exists(os.path.join(self.cfg.paths.data_folder, self.cfg.paths.h5_file)) \
            or not os.path.exists(os.path.join(self.cfg.paths.data_folder, self.cfg.paths.h5_file))):
            flag_download = True

        if flag_download:
            self._download_files()

    def _download_files(self):

        http = urllib3.PoolManager()
        for url in config.__urls__:
            print('Downloading ' + url + ' ...')
            response = http.request('GET', url)
            filename = os.path.basename(url)
            with open(os.path.join(self.cfg.paths.data_folder, filename), 'wb') as f:
                f.write(response.data)
            response.release_conn()
            with zipfile.ZipFile(os.path.join(self.cfg.paths.data_folder, filename), 'r') as f:
                f.extractall()


class trainset(Dataset):

    def __init__(self, dataset, ratio=None):

        self.dataset = dataset
        self.ratio = ratio or self.dataset.cfg.data_ratio

    def __getitem__(self, index):

        return self.dataset[index]

    def __len__(self):

        return math.floor(len(self.dataset) * self.ratio)


class testset(Dataset):

    def __init__(self, dataset, ratio=None):

        self.dataset = dataset
        self.ratio = ratio or (1 - self.dataset.cfg.data_ratio)

    def __getitem__(self, index):

        return self.dataset[-index-1]

    def __len__(self):

        return math.ceil(len(self.dataset) * self.ratio)


if __name__ == "__main__":

    gaze = Gaze()

    train_set = trainset(gaze)
    test_set = testset(train_set.dataset)

    for data, label in train_set:
        print(data.shape)
        print(label)

    for data, label in test_set:
        print(data.shape)
        print(label)