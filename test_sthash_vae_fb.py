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
FILE = './save/MNIST/model_sthash_vae_epoch_100_exp6.pth'
FILE_fb_sthash_vae = './save/MNIST/fb_sthash_vae_epoch_100_exp6.pth'
#FILE_fb_sthash = './save/MNIST/fb_novae_epoch_50.pth'
FILE_fb_sthash = './save/MNIST/fb_sthash_buddy_epoch_100_exp6.pth'
# device
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

model = Unet(im_channels=1, enable_attention=False)

model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
filter_bank_sthash_vae = torch.load(FILE_fb_sthash_vae, map_location=device)
filter_bank_sthash = torch.load(FILE_fb_sthash, map_location=device)

'''
fb = filter_bank_sthash.squeeze(0)
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
        LR_images = F.interpolate(images, scale_factor=0.5, mode='bilinear')
        LR_images = F.interpolate(LR_images, scale_factor=2.0, mode='bilinear')
        LR_images = LR_images.to(device)
        # unfold to patches
        unfold = torch.nn.Unfold(kernel_size=(7, 7), padding=(3, 3))
        LR_images_unfolded = unfold(LR_images)
        LR_images_unfolded_input = torch.permute(LR_images_unfolded, (0, 2, 1))
        LR_images_unfolded_input = LR_images_unfolded_input.reshape(-1, 1, 7, 7)

        sthash_filters = hash_and_get_filter(LR_images, filter_bank_sthash, training=False)
        pred_mean, pred_variance, recon_out, ker_pred, filter_out = model(LR_images_unfolded_input, filter_bank_sthash_vae, sthash_filters)

        #ker_pred = ker_pred.view(100, -1, 49)
        #ker_pred = ker_pred.permute(0, 2, 1)
        # perform filtering
        #filtered = torch.linalg.vecdot(LR_images_unfolded, ker_pred, dim=1)
        #filtered = filtered.view(100, -1, 28, 28)

        filtered = filter_out.view(batch_size, -1, 28, 28)

        fold = torch.nn.Fold(output_size=(28, 28), kernel_size=(7, 7), padding=(3, 3))
        ones = torch.ones_like(LR_images_unfolded)
        overlap_count = fold(ones)
        recon_out = recon_out.view(batch_size, -1, 49)
        recon_out = recon_out.permute(0, 2, 1)
        folded_recon = fold(recon_out)
        folded_recon = folded_recon / overlap_count

        folded_recon = folded_recon.to('cpu')
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
            plt.subplot(8, 3, idx + 1)
            tmp = images[idx, :, :, :]
            tmp = torch.permute(tmp, (1, 2, 0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(8, 3, idx + 1 + 6)
            tmp = LR_images[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(8, 3, idx + 1 + 12)
            tmp = filtered[idx, :, :, :]
            tmp = torch.permute(tmp, (1, 2, 0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(8, 3, idx + 1 + 18)
            tmp = folded_recon[idx, :, :, :]
            tmp = torch.permute(tmp, (1, 2, 0))
            plt.imshow(tmp, cmap='gray')
        plt.show()
        '''

        print(f'{idx} ', end='')
        if idx % 50 == 0:
            print('')

    avg_psnr = avg_psnr / idx
    print(f'PSNR: {avg_psnr}')






