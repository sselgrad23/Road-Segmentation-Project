'''
This file can load a generator weights file and generate the corresponding segmentations for a given dataset.
'''

from os import makedirs
import argparse
import data
import v2GAN
import torch
from torch.utils.tensorboard import SummaryWriter
import imageio.v3 as iio




parser = argparse.ArgumentParser(description='Train Barlow Twin Predictability Minimization')
parser.add_argument('--eval_data', type=str, default='/hdd/data/eth_road_segmentation_data/test', help='location of evaluation data')
parser.add_argument('--generator_weights_file', type=str, help='location of generator weights file')
parser.add_argument('--batch_size', default=4, type=int, help='number of epochs to train for')
parser.add_argument('--save_dir', default='save/', type=str, help='save dir')
parser.add_argument('--workers', default=20, type=int, help='Number of CPUs')


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def evaluate(generator, eval_data_loader):

    generator.eval()

    evaluated_images = []

    with torch.no_grad():
        for batch in eval_data_loader:
            segmentations = generator(batch.to(device)).cpu()
            for segmentation in segmentations:
                # get rid of batch dimension
                segmentation = segmentation.squeeze().numpy()
                evaluated_images.append(segmentation)

    return evaluated_images


def main():
    args = parser.parse_args()

    eval_data = data.ETHEvalData(args.data)

    generator = v2GAN.Generator().to(device)

    generator.load_state_dict(torch.load(args.generator_weights_file))
    generator.eval()

    makedirs(args.save_dir, exist_ok=True)

    # TODO make sure that the DataLoader does not reorder anything
    eval_data_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    evaluated_images = evaluate(generator, eval_data_loader)

    for idx, evaluated_image in enumerate(evaluated_images):
        # manually threshold (# TODO are these values good)
        # TODO do we want additional post processing?
        evaluated_image[evaluated_image >= 0.5] = 1
        evaluated_image[evaluated_image < 0.5] = 0
        # map from [0,1] to [0,255] as well as float to int
        evaluated_image = evaluated_image * 255
        evaluated_image = evaluated_image.astype('uint8')
        iio.imwrite(f'seg_{idx}.png',evaluated_image, plugin="pillow", extension=".png")


if __name__ == '__main__':
    main()
