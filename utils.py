import torch
import matplotlib.pyplot as plt
import imageio.v3 as iio
import torch.nn as nn
from torchmetrics import F1Score, Recall, Precision, Dice
import tqdm
from torch.utils.tensorboard import SummaryWriter


def save_gan(gan, dis, opt_gan, opt_dis, epoch, path):
    state = {
        'epoch': epoch,
        'gan': gan.state_dict(),
        'dis': dis.state_dict(),
        'opt_gan': opt_gan.state_dict(),
        'opt_dis': opt_dis.state_dict()
    }
    torch.save(state, path)


def load_gan(gan, dis, opt_gan, opt_disc, path):
    state = torch.load(path)
    gan.load_state_dict(state['gan'])
    dis.load_state_dict(state['dis'])
    opt_gan.load_state_dict(state['opt_gan'])
    opt_disc.load_state_dict(state['opt_dis'])


def save_model(model, opt, epoch, path):
    state = {
        'epoch': epoch,
        'gan': model.state_dict(),
        'opt': opt.state_dict(),
    }
    torch.save(state, path)


def load_model(model, opt, path):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['opt'])


def evaluate_train(generator, eval_data_loader, eval_dir, epoch, device, writer, sigmoid=False):
    generator.eval()
    img_ctr = 0
    sig = nn.Sigmoid()
    with torch.no_grad():
        thresholded_segmentation_list = []
        labels_list = []
        for sat, labels in tqdm.tqdm(eval_data_loader):
            if sigmoid:
                segmentations = sig(generator(sat.to(device))).cpu()
            else:
                segmentations = generator(sat.to(device)).cpu()

            labels = labels.cpu()

            for idx, segmentation in enumerate(segmentations):
                # get rid of batch dimension
                # segmentation = segmentation.squeeze()
                segmentation_thresholded = segmentation
                # map from [0,1] to [0,255] as well as float to int
                segmentation_thresholded[segmentation_thresholded >= 0.5] = 1
                segmentation_thresholded[segmentation_thresholded < 0.5] = 0

                #if img_ctr in [0, 1]:
                    # print(f"img:   {out[0]}")
                    # print(segmentation_thresholded)
                    # print(segmentation)

                    #writer.add_image(f'train_img_thresh_{img_ctr}', segmentation_thresholded.squeeze(), dataformats='HW', global_step=epoch)
                writer.add_image(f'train_img_{img_ctr}', segmentation.squeeze(), dataformats='HW', global_step=epoch)


                segmentation = segmentation * 255
                segmentation = segmentation.int()
                thresholded_segmentation_list.append(segmentation_thresholded.int())
                segmentation_thresholded = segmentation_thresholded * 255
                segmentation_thresholded = segmentation_thresholded.int()
                labels_list.append(labels[idx].int())
                if eval_dir:
                    # iio.imwrite(f'{eval_dir}/seg_thresh_{img_ctr}_{epoch}.png', segmentation_thresholded, plugin="pillow", extension=".png")
                    iio.imwrite(f'{eval_dir}/seg_{img_ctr}_{epoch}.png', segmentation, plugin="pillow",
                                extension=".png")

                img_ctr += 1
        # print(thresholded_segmentation_list)
        # print(thresholded_segmentation_list)

        stats = get_stats(torch.stack(thresholded_segmentation_list), torch.stack(labels_list))

        for key, value in stats.items():
            writer.add_scalar(key, value, epoch)




def evaluate(generator, eval_data_loader, eval_dir, epoch, device, writer, sigmoid=False):
    generator.eval()
    img_ctr = 0
    sig = nn.Sigmoid()
    with torch.no_grad():
        thresholded_segmentation_list = []
        labels_list = []
        for sat, labels in tqdm.tqdm(eval_data_loader):
            if sigmoid:
                segmentations = sig(generator(sat.to(device))).cpu()
            else:
                segmentations = generator(sat.to(device)).cpu()

            labels = labels.cpu()

            for idx, segmentation in enumerate(segmentations):
                # get rid of batch dimension
                # segmentation = segmentation.squeeze()
                segmentation_thresholded = segmentation
                # map from [0,1] to [0,255] as well as float to int
                segmentation_thresholded[segmentation_thresholded >= 0.5] = 1
                segmentation_thresholded[segmentation_thresholded < 0.5] = 0
                segmentation = segmentation * 255
                segmentation = segmentation.int()
                thresholded_segmentation_list.append(segmentation_thresholded.int())
                segmentation_thresholded = segmentation_thresholded * 255
                segmentation_thresholded = segmentation_thresholded.int()
                labels_list.append(labels[idx].int())
                if eval_dir:
                    iio.imwrite(f'{eval_dir}/eval_seg_thresh_{img_ctr}_{epoch}.png', segmentation_thresholded,
                                plugin="pillow", extension=".png")
                    iio.imwrite(f'{eval_dir}/eval_seg_{img_ctr}_{epoch}.png', segmentation, plugin="pillow",
                                extension=".png")
                if img_ctr in [0, 1]:
                    # print(f"img:   {out[0]}")
                    segmentation = segmentation.squeeze()
                    segmentation_thresholded = segmentation_thresholded.squeeze()
                    writer.add_image(f'train_img_thresh_{img_ctr}', segmentation_thresholded, dataformats='HW', global_step=epoch)
                    writer.add_image(f'train_img_{img_ctr}', segmentation, dataformats='HW', global_step=epoch)
                img_ctr += 1




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


f1 = F1Score(num_classes=1, multiclass=False)
recall = Recall(num_classes=1, multiclass=False)
precision = Precision(num_classes=1, multiclass=False)
dice = Dice(num_classes=1, multiclass=False)



# https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/script
# PyTroch version

SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    # return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    return thresholded.mean()

def get_stats(generated_segmentation, real_segmentation):
    stats = {
        'f1': f1(generated_segmentation, real_segmentation),
        'recall': recall(generated_segmentation, real_segmentation),
        'precision': precision(generated_segmentation, real_segmentation),
        'dice': dice(generated_segmentation, real_segmentation),
        'iou': iou_pytorch(generated_segmentation, real_segmentation)
    }
    return stats



def peek_data(data_loader):


    images, segmentations = next(iter(data_loader))
    assert images.shape[0] >= 4, 'Batch size needs to be at least 4 for --data_peek to work.'
    print(f'Min: {torch.min(images)}, Max: {torch.max(images)}, Mean: {torch.mean(images)}')
    plt.hist(images.flatten(), bins=20)
    plt.show()

    images = (images / 10) + 0.5

    f, axarr = plt.subplots(2,4)

    axarr[0][0].imshow(images[0].permute(1, 2, 0))
    axarr[1][0].imshow(segmentations[0])
    axarr[0][1].imshow(images[1].permute(1, 2, 0))
    axarr[1][1].imshow(segmentations[1])
    axarr[0][2].imshow(images[2].permute(1, 2, 0))
    axarr[1][2].imshow(segmentations[2])
    axarr[0][3].imshow(images[3].permute(1, 2, 0))
    axarr[1][3].imshow(segmentations[3])

    plt.show()
