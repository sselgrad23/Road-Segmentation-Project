import torch.nn
import tqdm


loss = torch.nn.BCEWithLogitsLoss()

def train_epoch(model, optimizer, train_data_loader, epoch, device, writer):

    model.train()

    train_loader_length = len(train_data_loader)

    epoch_loss = 0
    for i, (satellite_image, segmentation) in enumerate(tqdm.tqdm(train_data_loader)):

        satellite_image = satellite_image.to(device)
        segmentation = segmentation.to(device)
        segmentation = segmentation.unsqueeze(dim=1)

        out = model(satellite_image)
        # print(torch.max(out), torch.min(out))
        #writer.add_scalar('max', torch.max(out), epoch*train_loader_length + i)
        #writer.add_scalar('min', torch.min(out), epoch*train_loader_length + i)
        writer.add_scalars('min_max', {'min': torch.min(out), 'max': torch.max(out)}, epoch*train_loader_length + i)

        # print(out)

        l = loss(out, segmentation)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += l.item()



    writer.add_scalar('train_loss_epoch_sum', epoch_loss, epoch)


