
import tqdm
import torch
from torch import nn

import losses

sig = nn.Sigmoid()
bce_logits_loss = torch.nn.BCEWithLogitsLoss()
bce_loss = torch.nn.BCELoss()
dice_loss = losses.DiceLoss()

def generator_loss_fn(input):
    lbl = torch.ones(input.shape, dtype=torch.float32, device=input.device)
    return bce_logits_loss(input, lbl)


def train_gan_epoch(generator, discriminator, optimizer_generator, optimizer_discriminator, train_data_loader, lmbda,
                    epoch, device, writer, args, sigmoid=False):

    # seg_loss = losses.DiceLoss()

    real_segmentation_label = torch.tensor([1.0]).to(device)
    generated_segmentation_label = torch.tensor([0.0]).to(device)

    generator.train()
    discriminator.train()

    tot_gen = 0
    tot_dis = 0
    tot_seg = 0
    tot_generator_loss = 0

    for i, (satellite_image, segmentation) in enumerate(tqdm.tqdm(train_data_loader, 0)):
        optimizer_discriminator.zero_grad()
        # copy onto correct device
        satellite_image = satellite_image.to(device)
        segmentation = segmentation.to(device)
        segmentation = segmentation.unsqueeze(dim=1)

        # DISCRIMINATOR TRAINING

        with torch.no_grad():
            generated_segmentation = generator(satellite_image)

        err_discriminator_real = discriminator(segmentation, satellite_image)

        err_discriminator_generated = discriminator(generated_segmentation, satellite_image)

        discriminator_loss = bce_logits_loss(err_discriminator_real, real_segmentation_label.repeat(err_discriminator_real.shape[0], 1))\
                             + bce_logits_loss(err_discriminator_generated, generated_segmentation_label.repeat(err_discriminator_generated.shape[0], 1))

        discriminator_loss.backward()

        optimizer_discriminator.step()

        # GENERATOR TRAINING

        optimizer_generator.zero_grad()

        generated_segmentation = generator(satellite_image)

        err_discriminator_generated = discriminator(generated_segmentation, satellite_image)

        if args.loss == 'dice':
            segmentation_loss = dice_loss(sig(generated_segmentation, segmentation)) if sigmoid else dice_loss(generated_segmentation, segmentation)
        else:
            segmentation_loss = bce_logits_loss(generated_segmentation, segmentation) if sigmoid else bce_loss(generated_segmentation, segmentation)

        generator_loss = (1 - lmbda) * generator_loss_fn(err_discriminator_generated) + lmbda * segmentation_loss

        generator_loss.backward()
        optimizer_generator.step()

        tot_gen += generator_loss.item()
        tot_dis += discriminator_loss.item()
        tot_seg += segmentation_loss.item()
        tot_generator_loss += generator_loss.item()

    writer.add_scalar('generator_losses', tot_gen, epoch)
    writer.add_scalar('discriminator_losses', tot_dis, epoch)
    writer.add_scalar('segmentation_loss', tot_seg, epoch)
    writer.add_scalar('gen_seg_loss', tot_generator_loss, epoch)


