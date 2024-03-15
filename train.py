import tqdm
import torch

from torch.utils.tensorboard import SummaryWriter

import losses


def generator_loss_fn(input):
    return - torch.log(input.clamp(min=1e-8)).mean()


def train_gan_epoch(generator, discriminator, optimizer_generator, optimizer_discriminator, train_data_loader, lmbda, epoch, device, writer):
    loss = torch.nn.BCELoss()
    seg_loss = losses.DiceLoss()

    real_segmentation_label = torch.tensor([1.0]).to(device)
    generated_segmentation_label = torch.tensor([0.0]).to(device)

    generator.train()
    discriminator.train()

    tot_gen = 0
    tot_dis = 0
    tot_seg = 0
    tot_gen_seg = 0

    for i, (satellite_image, segmentation) in enumerate(tqdm.tqdm(train_data_loader, 0)):

        satellite_image = satellite_image.to(device)
        segmentation = segmentation.to(device)
        segmentation = segmentation.unsqueeze(dim=1)

        # DISCRIMINATOR TRAINING

        optimizer_discriminator.zero_grad()

        with torch.no_grad():
            generated_segmentation = generator(satellite_image)

        err_discriminator_real = discriminator(segmentation, satellite_image).view(-1)
        
        real_label = real_segmentation_label.repeat(err_discriminator_real.shape[0],)
        fake_label = generated_segmentation_label.repeat(err_discriminator_real.shape[0],)

        err_discriminator_generated = discriminator(generated_segmentation, satellite_image).view(-1)

        discriminator_loss = loss(err_discriminator_real, real_label) + loss(err_discriminator_generated, fake_label)

        discriminator_loss.backward()

        optimizer_discriminator.step()
        

        # GENERATOR TRAINING
        optimizer_generator.zero_grad()

        generated_segmentation = generator(satellite_image)

        err_discriminator_generated = discriminator(generated_segmentation, satellite_image).view(-1)

        segmentation_loss = seg_loss(generated_segmentation, segmentation)
        
        generator_loss = loss(err_discriminator_generated, real_label)

        # lambda weighs the losses against each other
        gen_seg_loss = (1 - lmbda) * generator_loss + lmbda * segmentation_loss
        

        gen_seg_loss.backward()
        optimizer_generator.step()
        
        tot_gen += generator_loss.item()
        tot_dis += discriminator_loss.item()
        tot_seg += segmentation_loss.item()
        tot_gen_seg += gen_seg_loss.item()

    writer.add_scalar('generator_losses', tot_gen, epoch)
    writer.add_scalar('discriminator_losses', tot_dis, epoch)
    writer.add_scalar('segmentation_loss', tot_seg, epoch)
    writer.add_scalar('gen_seg_loss', tot_gen_seg, epoch)
    
def train_model_epoch(model, opt, train_data_loader, epoch, device, writer):
    loss = losses.DiceLoss()

    model.train()

    tot_seg = 0

    for i, (satellite_image, segmentation) in enumerate(tqdm.tqdm(train_data_loader, 0)):

        # copy onto correct device
        satellite_image = satellite_image.to(device)
        segmentation = segmentation.to(device)
        segmentation = segmentation.unsqueeze(dim=1)

        opt.zero_grad()

        pred_seg = model(satellite_image)

        seg_loss = loss(pred_seg, segmentation)

        seg_loss.backward()
        opt.step()
        
        tot_seg += seg_loss.item()

    writer.add_scalar('segmentation_loss', tot_seg, epoch)

