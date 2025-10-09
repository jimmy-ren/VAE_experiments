import sys
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.video_utils import unfold

from main_network import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image

batch_size = 100
FILE = './save/MNIST/model_vae_epoch_510.pth'
FILE_fb = './save/MNIST/fb_vae_epoch_510.pth'
# device
device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

model = Unet(im_channels=1, enable_attention=False)

model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
filter_bank = torch.load(FILE_fb, map_location=device)
#filter_bank = torch.zeros(size=(1,49,8,3,3), device=device)
#filter_bank[:,24:25,:,:,:] = 1
'''
filter_bank.requires_grad = False
fb = filter_bank.squeeze(0)
fb = fb.permute(1, 2, 3, 0)
fb = fb.to('cpu').detach()
#fb = torch.softmax(fb, dim=-1)
# visualize the filter bank
for idx1 in range(3):
    for idx2 in range(3):
        for idx3 in range(8):
            plt.subplot(9, 8, 24*idx1+(8*idx2+idx3+1))
            k = fb[idx3, idx1, idx2, :]
            k = k.reshape([7, 7])
            #tmp = torch.permute(tmp, (1, 2, 0))
            plt.imshow(k, cmap='gray')
plt.show()
'''

# model parameters count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.eval()
model.to(device)

t = transforms.Compose([
    transforms.ToTensor(),
])
# MNIST dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=t,
                                          download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    print('testing...')
    avg_psnr = 0.0
    for idx, (images, labels) in enumerate(test_loader, start=1):
        '''
        for idx in range(6):
            plt.subplot(2, 3, idx+1)
            tmp = images[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')
        plt.show()
        '''

        LR_images = F.interpolate(images, scale_factor=0.5, mode='bilinear')
        LR_images = F.interpolate(LR_images, scale_factor=2.0, mode='bilinear')
        LR_images = LR_images.to(device)
        # unfold to patches
        unfold = torch.nn.Unfold(kernel_size=(7, 7), padding=(3, 3))
        LR_images_unfolded = unfold(LR_images)
        LR_images_unfolded_input = torch.permute(LR_images_unfolded, (0, 2, 1))
        LR_images_unfolded_input = LR_images_unfolded_input.reshape(-1, 1, 7, 7)

        pred_mean, pred_variance, outputs, ker_pred, filter_out = model(LR_images_unfolded_input, filter_bank)

        ker_pred = ker_pred.view(100, -1, 49)
        ker_pred = ker_pred.permute(0, 2, 1)
        # perform filtering
        filtered = torch.linalg.vecdot(LR_images_unfolded, ker_pred, dim=1)
        filtered = filtered.view(100, -1, 28, 28)

        LR_images = torch.clip(LR_images.to('cpu'), min=0.0, max=1.0)
        filtered = torch.clip(filtered.to('cpu'), min=0.0, max=1.0)
        '''
        images = images.numpy()
        
        im1 = filtered[1,0,:,:]
        im2 = images[1,0,:,:]
        im3 = LR_images[1,0,:,:]

        p = calculate_psnr(im1, im2, 1.0)
        p2 = calculate_psnr(im3, im2, 1.0)
        
        psnr_ = batch_psnr(LR_images, images)
        '''
        psnr = batch_psnr(filtered, images)
        avg_psnr = avg_psnr + psnr

        '''
        for idx in range(6):
            plt.subplot(6, 3, idx + 1)
            tmp = images[idx, :, :, :]
            tmp = torch.permute(tmp, (1, 2, 0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(6, 3, idx + 1 + 6)
            tmp = LR_images[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(6, 3, idx + 1 + 12)
            tmp = filtered[idx, :, :, :]
            tmp = torch.permute(tmp, (1, 2, 0))
            plt.imshow(tmp, cmap='gray')
        plt.show()
        '''

        print(f'{idx} ', end='')
        if idx % 50 == 0:
            print('')

    avg_psnr = avg_psnr / idx
    print(f'PSNR: {avg_psnr}')






