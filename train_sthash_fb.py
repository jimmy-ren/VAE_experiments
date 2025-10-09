import math
from aux_funs import *
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from main_network import*

#some config
num_epochs = 50
batch_size = 100
learning_rate = 0.0002
center_crop_size = (24, 24)
crop_size = (16, 16)

t = transforms.Compose([
        transforms.CenterCrop(center_crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=crop_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5), (0.5)),
    ])
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=t,
                                          download=True)

# device
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

filter_bank = torch.normal(size=(1,49,8,3,3), mean=0.0, std=1/100, requires_grad=True, device=device)

# loss and optimizer
criterion = nn.MSELoss()
fb_optimizer = torch.optim.Adam([filter_bank], lr=learning_rate)

n_total_steps = len(train_loader)
loss_sum_in_step_loop = 0
loss_sum_in_epoch_loop = 0
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, c, h, w]
        LR_images = F.interpolate(images, scale_factor=0.5, mode='bilinear')
        LR_images = F.interpolate(LR_images, size=crop_size, mode='bilinear')

        LR_images = LR_images.to(device)
        images = images.to(device)

        final_filters = hash_and_get_filter(LR_images, filter_bank)

        unfold = torch.nn.Unfold(kernel_size=(7, 7))
        LR_images_unfolded = unfold(LR_images)
        LR_images_unfolded = torch.permute(LR_images_unfolded, (0, 2, 1))
        LR_images_unfolded = LR_images_unfolded.reshape(-1, 1, 7, 7)
        input_img = LR_images_unfolded.view(-1, 49)
        filter_out = torch.linalg.vecdot(input_img, final_filters, dim=1)

        images_unfolded = unfold(images)
        images_unfolded = torch.permute(images_unfolded, (0, 2, 1))
        images_unfolded = images_unfolded.reshape(-1, 1, 7, 7)

        img_GT = images_unfolded[:, :, 3:4, 3:4]
        img_GT = img_GT.view(filter_out.shape)

        loss = criterion(filter_out, img_GT)

        loss.backward()
        fb_optimizer.step()
        fb_optimizer.zero_grad()

        FILE_fb = f'./save/MNIST/fb_novae_epoch_{epoch + 1}.pth'

        loss_sum_in_step_loop += loss.item()
        loss_sum_in_epoch_loop += loss.item()
        if (i + 1) % 100 == 0:
            loss_in_step_loop = loss_sum_in_step_loop / 100
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, avg loss={loss_in_step_loop:.4f}')
            loss_sum_in_step_loop = 0


    if (epoch + 1) % 1 == 0:
        '''
        LR_images_unfolded = LR_images_unfolded.to('cpu')
        final_filters = final_filters.to('cpu').detach()
        for idx in range(6):
            plt.subplot(2*3, 3, idx+1)
            tmp = LR_images_unfolded[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(2*3, 3, idx+1+6)
            tmp2 = final_filters[idx,:]
            tmp2 = tmp2.view(7, 7)
            plt.imshow(tmp2, cmap='gray')

        plt.show()
        '''
        fb = filter_bank.squeeze(0)
        fb = fb.permute(1, 2, 3, 0)
        fb = fb.to('cpu').detach()
        # fb = torch.softmax(fb, dim=-1)
        # visualize the filter bank
        counter = 0
        for idx1 in range(3):
            for idx2 in range(3):
                for idx3 in range(8):
                    counter += 1
                    plt.subplot(9, 8, counter)
                    k = fb[idx3, idx1, idx2, :]
                    k = k.reshape([7, 7])
                    # tmp = torch.permute(tmp, (1, 2, 0))
                    plt.axis('off')
                    plt.imshow(k, cmap='gray')
        plt.show()

    if (epoch + 1) % 10 == 0:
        torch.save(filter_bank, FILE_fb)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    loss_in_epoch_loop = loss_sum_in_epoch_loop / (i + 1)

    print(f'Elapsed time for one epoch: {elapsed_time:.6f} seconds, epoch avg loss={loss_in_epoch_loop:.4f}')
    loss_sum_in_epoch_loop = 0
