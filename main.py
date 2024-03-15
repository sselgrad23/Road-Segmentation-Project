###
# Uses code from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_div/wgan_div.py
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
import random
from os import makedirs
import argparse
from pathlib import Path

import YNet
import data
import models.v2GAN as v2GAN
import trainYNet
import trainv2GAN
from models.munet_gan_model import Generator, Discriminator
import torch
from torch.utils.tensorboard import SummaryWriter

from os import environ

from utils import evaluate_train, peek_data, evaluate

from train import train_gan_epoch, train_model_epoch

parser = argparse.ArgumentParser(description='Train Barlow Twin Predictability Minimization')
# lines taken from https://github.com/yaohungt/Barlow-Twins-HSIC/blob/main/linear.py
parser.add_argument('--model', type=str, required=True, help='location of data')
parser.add_argument('--data_dir', type=str, default='./cil_data/', help='location of data')
parser.add_argument('--epochs', default=10000, type=int, help='number of epochs to train for')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--workers', default=2, type=int, help='Number of CPUs')
parser.add_argument('--beta1', default=0.5, type=float, help='beta1 of adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of adam')
parser.add_argument('--eval_dir', type=str, help='eval dir')
parser.add_argument('--save_dir', default='save/', type=str, help='save dir')
parser.add_argument('--lmbda', default=0.5, type=float, help='lambda weighing parameter')
parser.add_argument('--lr', default=0.0003, type=float, help='adam learning rate')
parser.add_argument('--weight_decay', default=0.0, type=float, help='adam weight decay')
parser.add_argument('--checkpointing', action='store_true', help='If models should be checkpointed while training')
parser.add_argument('--loss', type=str, choices=['dice', 'bce'], default=['bce'], help='which segmentation loss to use')


# these control augmentation
parser.add_argument('--rotate', action='store_true', help='adds geometric.rotate.Rotate() to transforms')
parser.add_argument('--img_dim', type=int, default=400,
                    help='image dimensions for RandomCrop(width=img_dim , height=img_dim)')
parser.add_argument('--h_flip', action='store_true', help='adds HorizontalFlip() to transforms')
parser.add_argument('--v_flip', action='store_true', help='adds VerticalFlip() to transforms')
parser.add_argument('--brightness', type=float, default=None, help='set brightness limit to brightness')
parser.add_argument('--contrast', type=float, default=None, help='set contrast limit to contrast')
parser.add_argument('--distort', action='store_true', help='wether to distort the images')

parser.add_argument('--val_period', type=int, default=1, help='per how many epochs a validation should be done')
parser.add_argument('--peek_data', action='store_true', help='get popup of preprocessed pictures for visual inspection')
parser.add_argument('--pretrain_finetune', action='store_true',
                    help='if there should be a pretrain dataset for non ethcil data and a finetune eth cil dataset')

parser.add_argument('--train_data', nargs='+', choices=['cil', 'berlin', 'paris', 'zurich', 'chicago'], default=['cil'],
                    help='list of data sets for training')
parser.add_argument('--val_data', nargs='+', choices=['cil', 'berlin', 'paris', 'zurich', 'chicago'], default=['cil'],
                    help='list of data sets for evaluation')
parser.add_argument('--ynet_version', default=1, type=int, help='which ynet version to use')

# this allows for easier tracking in tensorboard, if we are on an lsf cluster populate the comment with the jobid.
if environ.get('LSB_JOBID') is not None:
    tracking_code = 'l-' + environ.get('LSB_JOBID')
else:
    tracking_code = 'r-' + str(random.randint(0, 10000000000))

writer = SummaryWriter(comment=f"-{tracking_code}")

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'


def main():
    args = parser.parse_args()

    print(args)
    print(f'Tensorboard Tracking Code is {tracking_code}')

    current_model = args.model
    train_data_loader, val_data_loader = data.create_dataloaders(args)

    if args.eval_dir:
        makedirs(args.eval_dir, exist_ok=True)

    makedirs(args.save_dir, exist_ok=True)

    if args.peek_data:
        peek_data(train_data_loader)

    if current_model == 'model':
        model = Generator().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                betas=(args.beta1, args.beta2))
    elif current_model == 'gan':
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    elif current_model == 'v2GAN':
        generator = v2GAN.Generator().to(device)
        discriminator = v2GAN.Discriminator().to(device)

        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.lr,
                                               betas=(args.beta1, args.beta2))

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                                                   betas=(args.beta1, args.beta2))

    elif current_model == 'YNet':
        model = YNet.YNet(400, 3, version=args.ynet_version).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                      betas=(args.beta1, args.beta2))

    elif current_model == 'YNetGAN':
        generator = YNet.YNet(400, 3, version=args.ynet_version).to(device)

        discriminator = v2GAN.Discriminator().to(device)
        # TODO confiugure
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                               betas=(args.beta1, args.beta2))

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                                                   betas=(args.beta1, args.beta2))

    print("Starting Training Loop...")

    for epoch in range(1, args.epochs):

        if current_model == 'model':
            train_model_epoch(model, opt, train_data_loader, epoch, device, writer)
        elif current_model == 'gan':
            train_gan_epoch(generator, discriminator, optimizer_generator, optimizer_discriminator, train_data_loader,
                            args.lmbda, epoch, device, writer)

        elif current_model in ['v2GAN']:
            trainv2GAN.train_gan_epoch(generator, discriminator, optimizer_generator, optimizer_discriminator, train_data_loader,
                        args.lmbda, epoch, device, writer, args)

        elif current_model in ['YNetGAN']:
            trainv2GAN.train_gan_epoch(generator, discriminator, optimizer_generator, optimizer_discriminator, train_data_loader,
                        args.lmbda, epoch, device, writer, args, sigmoid=True)

        elif current_model == 'YNet':
            trainYNet.train_epoch(model, optimizer, train_data_loader, epoch, device, writer)

        if epoch % args.val_period == 0 or epoch == 1 and val_data_loader:
            if current_model in ['model', 'YNet']:
                evaluate_train(model, val_data_loader, args.eval_dir, epoch, device, writer, sigmoid=True)
            if current_model in ['YNetGAN']:
                evaluate_train(generator, val_data_loader, args.eval_dir, epoch, device, writer, sigmoid=True)
            elif current_model == 'gan' or current_model == 'v2GAN':
                evaluate_train(generator, val_data_loader, args.eval_dir, epoch, device, writer)

# this is dupl
        if epoch % args.val_period == 0 and args.save_dir and args.checkpointing:
            if current_model in ['YNetGAN','v2GAN', 'gan']:
                model = generator
            save_name = Path(args.save_dir) / f'model_{current_model}_{epoch}_{tracking_code}.pth'
            torch.save(model.state_dict(), save_name)

    if current_model in ['YNetGAN','v2GAN', 'gan']:
        model = generator
    save_name = Path(args.save_dir) / f'model_{current_model}_final_{tracking_code}.pth'
    torch.save(model.state_dict(), save_name)

if __name__ == '__main__':
    main()