import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.video_utils import unfold

from naive_kpn import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image

batch_size = 100
FILE = './save/MNIST/model_naive_kpn_epoch_200.pth'

# device
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

model = NaiveKPN(im_channels=1, enable_attention=False)
model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))

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

        ker_pred, filter_out = model(LR_images_unfolded_input)

        filtered = filter_out.view(batch_size, -1, 28, 28)

        LR_images = torch.clip(LR_images.to('cpu'), min=0.0, max=1.0)
        filtered = torch.clip(filtered.to('cpu'), min=0.0, max=1.0)

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

        plt.show()
        '''

        print(f'{idx} ', end='')
        if idx % 50 == 0:
            print('')

    avg_psnr = avg_psnr / idx
    print(f'PSNR: {avg_psnr}')






