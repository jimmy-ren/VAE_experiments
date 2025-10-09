import sys
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from main_network import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image

batch_size = 100
FILE = f'./save/MNIST/model_vae_epoch_100.pth'
FILE_fb = './save/MNIST/fb_vae_epoch_50.pth'

# device
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

model = Unet(im_channels=1, enable_attention=False)

model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
filter_bank = torch.load(FILE_fb, map_location=device)
filter_bank.requires_grad = False

# model parameters count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.eval()
model.to(device)
model.eval()

crop_size = (7, 7)
t = transforms.Compose([
    #transforms.Pad(padding=2, fill=0, padding_mode='constant'),
    transforms.RandomCrop(size=crop_size),
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5)),
])
# MNIST dataset, images are 1x28x28 pad to 1x32x32
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=t,
                                          download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

processed_data = [['x', 'y', 'label']]
with torch.no_grad():
    print('testing...')
    for images, labels in test_loader:
        '''
        for idx in range(6):
            plt.subplot(2, 3, idx+1)
            tmp = images[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')
        plt.show()
        '''
        images = images.to(device)
        pred_mean, pred_variance, outputs, ker_pred, filter_out = model(images, filter_bank)
        pred_mean = pred_mean.to('cpu')
        pred_variance = pred_variance.to('cpu')
        # write to a csv file in the form
        # mean dim1, mean dim2, label
        for i in range(pred_mean.shape[0]):
            processed_row = []
            processed_row.append(pred_mean[i,0].item())
            processed_row.append(pred_mean[i,1].item())
            processed_row.append(labels[i].item())
            processed_data.append(processed_row)

filename = "output.csv"

# Open the file in write mode ('w') with newline='' to prevent extra blank rows
with open(filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write each row of data
    csv_writer.writerows(processed_data)

print(f"Data written to {filename}")
