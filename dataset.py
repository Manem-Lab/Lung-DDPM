#-*- coding:utf-8 -*-

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import os

class CTPairImageGenerator(Dataset):
    def __init__(self,
            ct_path: str,
            mask_path: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 3,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_size = input_size
        self.depth_size = depth_size
        self.mask_path = mask_path
        self.ct_path = ct_path
        self.pair_files = self.pair_file()
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        mask_files = os.listdir(self.mask_path)
        pairs = []
        for mask_file in mask_files:
            patient_id = mask_file.split('.')[0]
            mask_file = os.path.join(self.mask_path, patient_id + '.nii.gz')
            ct_file = os.path.join(self.ct_path, patient_id + '.nii.gz')
            pairs.append((mask_file, ct_file))
        return pairs

    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape + ( self.input_channel - 1,))
        result_img[masked_img==LabelEnum.LUNG.value, 0] = 1
        result_img[masked_img==LabelEnum.Nodule.value, 1] = 1
        return result_img

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        if not pass_scaler:
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, self.input_channel-1))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        input_img = np.transpose(input_img, (1, 2, 0))
        target_img = self.read_image(target_file)
        target_img = np.transpose(target_img, (1, 2, 0))
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
        target_img = self.resize_img(target_img)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input':input_img, 'target':target_img}
