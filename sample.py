#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import create_model
from torchvision.transforms import Compose, Lambda
from utils.dtypes import LabelEnum
import nibabel as nib
import torchio as tio
import numpy as np
import argparse
import torch
import os
import glob
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--ct_path', type=str, default='data/LIDC-IDRI/CT')
parser.add_argument('--mask_path', type=str, default='data/LIDC-IDRI/SEG')
parser.add_argument('--sample_path', type=str, default="data/LIDC-IDRI/SAMPLE")
parser.add_argument('--vis_path', type=str, default="data/LIDC-IDRI/VIS")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=3)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--weightfile', type=str, default="checkpoints/Lung-DDPM-LIDC-IDRI-100000-steps.pt")
parser.add_argument('--blend_from', type=int, default=250)
args = parser.parse_args()

ct_path = args.ct_path
mask_path = args.mask_path
sample_path = args.sample_path
vis_path = args.vis_path
input_size = args.input_size
depth_size = args.depth_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
in_channels = args.num_class_labels
blend_from = args.blend_from
out_channels = 1


def save_gif(sampleImage, vis_path, type):
    gif_frames = []
    slice_3d = sampleImage
    slice_3d = ((slice_3d - slice_3d.min()) / (slice_3d.max() - slice_3d.min()) * 255).astype(np.uint8)
    if type == 'mask':
        slice_3d[slice_3d < 10] = 0
        slice_3d[slice_3d > 240] = 255
        slice_3d[(slice_3d <= 240) & (slice_3d >= 10)] = 127
    for i in range(slice_3d.shape[0]):
        slice_2d = slice_3d[i, :, :]
        gif_frames.append(Image.fromarray(slice_2d))

    if type == 'mask':
        gif_path = os.path.join(vis_path, f'{msk_name.split(".")[0]}_mask.gif')
    elif type == 'sample':
        gif_path = os.path.join(vis_path, f'{msk_name.split(".")[0]}_sample.gif')
    elif type == 'ct':
        gif_path = os.path.join(vis_path, f'{msk_name.split(".")[0]}_ct.gif')
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)


def resize_img(img):
    h, w, d = img.shape
    if h != input_size or w != input_size or d != depth_size:
        img = tio.ScalarImage(tensor=img[np.newaxis, ...])
        cop = tio.Resize((input_size, input_size, depth_size))
        img = np.asarray(cop(img))[0]
    return img


def resize_img_4d(input_img):
    h, w, d, c = input_img.shape
    result_img = np.zeros((input_size, input_size, depth_size, in_channels-1))
    if h != input_size or w != input_size or d != depth_size:
        for ch in range(c):
            buff = input_img.copy()[..., ch]
            img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
            cop = tio.Resize((input_size, input_size, depth_size))
            img = np.asarray(cop(img))[0]
            result_img[..., ch] += img
        return result_img
    else:
        return input_img

def label2masks(masked_img):
    result_img = np.zeros(masked_img.shape + (in_channels - 1,))
    result_img[masked_img==LabelEnum.LUNG.value, 0] = 1
    result_img[masked_img==LabelEnum.Nodule.value, 1] = 1
    return result_img


if __name__ == '__main__':
    input_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.permute(3, 0, 1, 2)),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(4, 2))
    ])

    ct_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(3, 1)),
    ])

    mask_list = sorted(glob.glob(f"{mask_path}/*.nii.gz"))
    ct_list = sorted(glob.glob(f"{ct_path}/*.nii.gz"))
    print(f'{len(mask_list)} cases waiting for sampling!')

    model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=args.timesteps,
        loss_type='L1',
        with_condition=True,
    ).cuda()
    diffusion.load_state_dict(torch.load(weightfile)['ema'])
    print("Lung-DDPM model loaded!")

    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)
    scaler = MinMaxScaler()

    for mask_file in tqdm(mask_list):
        mask_nifti = nib.load(mask_file)
        # msk_name = inputfile.split('/')[-1]  # For Linux user
        msk_name = mask_file.split('\\')[-1]  # For Windows user
        maskImg = mask_nifti.get_fdata()
        ct = nib.load(os.path.join(ct_path, msk_name))
        ct_arr = ct.get_fdata()
        ct_arr = np.transpose(ct_arr, (1, 2, 0))
        mask_arr = np.transpose(maskImg, (1, 2, 0))
        ct_arr = resize_img(ct_arr)
        ctImg = np.transpose(ct_arr, (2, 0, 1))
        mask = np.transpose(resize_img(mask_arr), (2, 0, 1))
        save_gif(mask, vis_path, type='mask')
        save_gif(ctImg, vis_path, type='ct')
        mask_arr = label2masks(mask_arr)
        mask_arr = resize_img_4d(mask_arr)
        condition_tensor = input_transform(mask_arr)
        condition_tensor = condition_tensor.cuda()
        ct_arr = scaler.fit_transform(ct_arr.reshape(-1, ctImg.shape[-1])).reshape(ctImg.shape)
        ct_tensor = ct_transform(ct_arr)
        ct_tensor = torch.unsqueeze(ct_tensor, dim=0)
        ct_tensor = ct_tensor.cuda()
        sampleImage = diffusion.sample_lung_ddpm(x_start=ct_tensor, condition_tensors=condition_tensor, blend_from=blend_from)
        sampleImage = sampleImage.cpu()
        sampleImage = sampleImage.numpy()
        sampleImage = np.squeeze(sampleImage)
        sampleImage = np.transpose(sampleImage, (0, 2, 1))
        nifti_img = nib.Nifti1Image(sampleImage, affine=mask_nifti.affine)
        nib.save(nifti_img, os.path.join(sample_path, f'{msk_name}'))
        save_gif(sampleImage, vis_path, type='sample')
        torch.cuda.empty_cache()
