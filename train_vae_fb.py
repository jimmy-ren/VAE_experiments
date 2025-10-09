import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from main_network import*

#some config
num_epochs = 1000
batch_size = 100
learning_rate = 0.0002
learning_rate_fb = 0.0001
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

model = Unet(im_channels=1, enable_attention=False)

FILE = './save/MNIST/model_vae_epoch_100_vae_warmup.pth'
model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
model = model.to(device)

FILE_fb = './save/MNIST/fb_vae_epoch_100_with_vae_warmup.pth'
#filter_bank = torch.normal(size=(1,49,8,3,3), mean=0.0, std=1/100, requires_grad=True, device=device)
filter_bank = torch.load(FILE_fb, map_location=device)

# freeze the parameters
#for param in model.parameters():
#    param.requires_grad = False

'''
# visualize the filter bank
fb = filter_bank.to('cpu').detach()
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
#print(filter_bank.is_leaf)

# loss and optimizer
criterion = nn.MSELoss()
criterion2 = KLD_VAE_Loss()
criterion3 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
fb_optimizer = torch.optim.Adam([filter_bank], lr=learning_rate_fb)

# training loop
loss_sum_in_step_loop1 = 0
loss_sum_in_step_loop2 = 0
loss_sum_in_step_loop3 = 0
loss_sum_in_epoch_loop1 = 0
loss_sum_in_epoch_loop2 = 0
loss_sum_in_epoch_loop3 = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, c, h, w]
        LR_images = F.interpolate(images, scale_factor=0.5, mode='bilinear')
        LR_images = F.interpolate(LR_images, size=crop_size, mode='bilinear')

        '''
        for i in range(6):
            plt.subplot(4, 3, i+1)
            tmp = LR_images[i,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(4, 3, i + 1 + 6)
            tmp = images[i, :, :, :]
            tmp = torch.permute(tmp, (1, 2, 0))
            plt.imshow(tmp, cmap='gray')
        plt.show()
        '''

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
        pred_mean, pred_variance, outputs, ker_pred, filter_out = model(LR_images_unfolded, filter_bank)
        img_GT = images_unfolded[:,:,3:4,3:4]
        img_GT = img_GT.view(filter_out.shape)

        # reconstruction loss
        loss1 = criterion(outputs, LR_images_unfolded)
        # KLD loss
        loss2 = criterion2(pred_mean, pred_variance)
        # KPN loss
        loss3 = criterion3(filter_out, img_GT)
        loss = loss1 + 0.00001*loss2 + loss3
        #loss = loss3
        FILE = f'./save/MNIST/model_vae_epoch_{epoch + 1}.pth'
        FILE_fb = f'./save/MNIST/fb_vae_epoch_{epoch + 1}.pth'

        # backward

        loss.backward()
        #grad_input, = torch.autograd.grad(loss, filter_bank)

        #print(filter_bank.is_leaf)
        #print(filter_bank.grad)

        #loss2.backward(retain_graph=True)
        #loss1.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        fb_optimizer.step()
        fb_optimizer.zero_grad()

        loss_sum_in_step_loop1 += loss1.item()
        loss_sum_in_step_loop2 += loss2.item()
        loss_sum_in_step_loop3 += loss3.item()
        loss_sum_in_epoch_loop1 += loss1.item()
        loss_sum_in_epoch_loop2 += loss2.item()
        loss_sum_in_epoch_loop3 += loss3.item()
        if (i + 1) % 100 == 0:
            loss_in_step_loop1 = loss_sum_in_step_loop1 / 100
            loss_in_step_loop2 = loss_sum_in_step_loop2 / 100
            loss_in_step_loop3 = loss_sum_in_step_loop3 / 100
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, avg loss1={loss_in_step_loop1:.4f}, avg loss2={loss_in_step_loop2:.4f}, avg loss3={loss_in_step_loop3:.4f}')
            loss_sum_in_step_loop1 = 0
            loss_sum_in_step_loop2 = 0
            loss_sum_in_step_loop3 = 0


    if (epoch + 1) % 20 == 0:
        # save checkpoint every 100 epochs
        #torch.save(model.state_dict(), FILE)
        LR_images_unfolded = LR_images_unfolded.to('cpu')
        outputs = outputs.to('cpu').detach()
        ker_pred = ker_pred.to('cpu').detach()
        for idx in range(6):
            plt.subplot(2*3, 3, idx+1)
            tmp = LR_images_unfolded[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(2*3, 3, idx+1+6)
            tmp1 = outputs[idx,:,:,:]
            tmp1 = torch.permute(tmp1,(1,2,0))
            plt.imshow(tmp1, cmap='gray')

            plt.subplot(2*3, 3, idx+1+6+6)
            tmp2 = ker_pred[idx,:]
            tmp2 = tmp2.view(7, 7)
            plt.imshow(tmp2, cmap='gray')

        plt.show()


    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), FILE)
        torch.save(filter_bank, FILE_fb)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    loss_in_epoch_loop1 = loss_sum_in_epoch_loop1 / (i + 1)
    loss_in_epoch_loop2 = loss_sum_in_epoch_loop2 / (i + 1)
    loss_in_epoch_loop3 = loss_sum_in_epoch_loop3 / (i + 1)
    print(f'Elapsed time for one epoch: {elapsed_time:.6f} seconds, epoch avg loss={loss_in_epoch_loop1+loss_in_epoch_loop2+loss_in_epoch_loop3:.4f}')
    loss_sum_in_epoch_loop1 = 0
    loss_sum_in_epoch_loop2 = 0
    loss_sum_in_epoch_loop3 = 0