import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from naive_kpn import*

#some config
num_epochs = 200
batch_size = 100
learning_rate = 0.0002
center_crop_size = (24, 24)
crop_size = (12, 12)

t = transforms.Compose([
        transforms.CenterCrop(center_crop_size),
        transforms.RandomCrop(size=crop_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5), (0.5)),
    ])
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=t,
                                          download=True)

# device
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = NaiveKPN(im_channels=1, enable_attention=False)
model = model.to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training loop
loss_sum_in_step_loop = 0
loss_sum_in_epoch_loop = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, c, h, w]
        LR_images = F.interpolate(images, scale_factor=0.5, mode='bilinear')
        LR_images = F.interpolate(LR_images, size=crop_size, mode='bilinear')


        LR_images = LR_images.to(device)
        images = images.to(device)

        # unfolding the data
        unfold = torch.nn.Unfold(kernel_size=(7, 7))
        LR_images_unfolded = unfold(LR_images)
        LR_images_unfolded = torch.permute(LR_images_unfolded, (0, 2, 1))
        LR_images_unfolded = LR_images_unfolded.reshape(-1, 1, 7, 7)
        images_unfolded = unfold(images)
        images_unfolded = torch.permute(images_unfolded, (0, 2, 1))
        images_unfolded = images_unfolded.reshape(-1, 1, 7, 7)

        # forward
        ker_pred, filter_out = model(LR_images_unfolded)
        img_GT = images_unfolded[:,:,3:4,3:4]
        img_GT = img_GT.view(filter_out.shape)

        # KPN loss
        loss = criterion(filter_out, img_GT)

        FILE = f'./save/MNIST/model_naive_kpn_epoch_{epoch + 1}.pth'


        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        loss_sum_in_step_loop += loss.item()
        loss_sum_in_epoch_loop += loss.item()
        if (i + 1) % 100 == 0:
            loss_in_step_loop = loss_sum_in_step_loop / 100
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, avg loss={loss_in_step_loop:.4f}')
            loss_sum_in_step_loop = 0


    if (epoch + 1) % 20 == 0:
        # save checkpoint every 100 epochs
        #torch.save(model.state_dict(), FILE)
        LR_images_unfolded = LR_images_unfolded.to('cpu')
        ker_pred = ker_pred.to('cpu').detach()
        for idx in range(6):
            plt.subplot(2*3, 3, idx+1)
            tmp = LR_images_unfolded[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(2*3, 3, idx+1+6)
            tmp2 = ker_pred[idx,:]
            tmp2 = tmp2.view(7, 7)
            plt.imshow(tmp2, cmap='gray')

        plt.show()


    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), FILE)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    loss_in_epoch_loop = loss_sum_in_epoch_loop / (i + 1)
    print(f'Elapsed time for one epoch: {elapsed_time:.6f} seconds, epoch avg loss={loss_in_epoch_loop:.4f}')
    loss_sum_in_epoch_loop = 0





