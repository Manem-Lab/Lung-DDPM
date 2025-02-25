#-*- coding:utf-8 -*-

from torchvision.transforms import Compose, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import CTPairImageGenerator
import argparse
import torch
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--ct_path', type=str, default='data/LIDC-IDRI/CT')
parser.add_argument('--mask_path', type=str, default='data/LIDC-IDRI/CT')
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=3)
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument( '--start_steps', type=int, default=0)
args = parser.parse_args()

ct_path = args.ct_path
mask_path = args.mask_path
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
resume_weight = args.resume_weight
train_lr = args.train_lr
start_steps = args.start_steps

if __name__ == '__main__':
    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(3, 1)),
    ])

    input_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.permute(3, 0, 1, 2)),
        Lambda(lambda t: t.transpose(3, 1)),
    ])

    in_channels = num_class_labels
    out_channels = 1

    dataset = CTPairImageGenerator(
        ct_path=ct_path,
        mask_path=mask_path,
        input_size=input_size,
        depth_size=depth_size,
        input_channel=in_channels,
        transform=input_transform,
        target_transform=transform,
        full_channel_mask=True,
    )

    print(f'{len(dataset)} samples loaded.')

    model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels, channel_mult="").cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=args.timesteps,
        loss_type='l1',
        channels=out_channels
    ).cuda()

    if len(resume_weight) > 0:
        weight = torch.load(resume_weight, map_location='cuda')
        diffusion.load_state_dict(weight['ema'])
        print(f"Pretrained model loaded. Resume training from {start_steps}th steps!")

    trainer = Trainer(
        diffusion,
        dataset,
        start_steps=start_steps,
        depth_size=depth_size,
        train_batch_size=args.batchsize,
        train_lr=train_lr,
        train_num_steps=args.epochs,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        save_and_sample_every=save_and_sample_every,
    )

    trainer.train()
