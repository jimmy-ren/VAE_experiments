import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from main_network import*

#some config
num_epochs = 100
batch_size = 100
learning_rate = 0.0001
FILE = './save/MNIST/test_model_vae_epoch_1.pth'

t = transforms.Compose([
        transforms.Pad(padding=2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
# MNIST dataset, images are 1x28x28 pad to 1x32x32
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=t,
                                          download=True)

# device
device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = Unet(im_channels=1, enable_attention=False)

#model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
model = model.to(device)

# loss and optimizer
criterion = nn.MSELoss()
criterion2 = KLD_VAE_Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
loss_sum_in_step_loop1 = 0
loss_sum_in_step_loop2 = 0
loss_sum_in_epoch_loop1 = 0
loss_sum_in_epoch_loop2 = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, c, h, w]
        image_batch = images
        #tmp = images[0,0,:,:]
        #image_batch = image_batch[:,0:1,:,:]
        #images = images[:,0:1,:,:]
        '''
        for i in range(6):
            plt.subplot(2, 3, i+1)
            tmp = images[i,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')
        plt.show()
        '''

        images = images.to(device)

        # forward
        pred_mean, pred_variance, outputs = model(images)

        loss1 = criterion(outputs, images)
        loss2 = criterion2(pred_mean, pred_variance)
        loss = loss1 + 0.0001*loss2
        FILE = f'./save/MNIST/model_vae_epoch_{epoch + 1}.pth'

        # backward

        loss.backward()
        #loss2.backward(retain_graph=True)
        #loss1.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum_in_step_loop1 += loss1.item()
        loss_sum_in_step_loop2 += loss2.item()
        loss_sum_in_epoch_loop1 += loss1.item()
        loss_sum_in_epoch_loop2 += loss2.item()
        if (i + 1) % 100 == 0:
            loss_in_step_loop1 = loss_sum_in_step_loop1 / 100
            loss_in_step_loop2 = loss_sum_in_step_loop2 / 100
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, avg loss1={loss_in_step_loop1:.4f}, avg loss2={loss_in_step_loop2:.4f}')
            loss_sum_in_step_loop1 = 0
            loss_sum_in_step_loop2 = 0

    if (epoch + 1) % 1 == 0:
        # save checkpoint every 100 epochs
        #torch.save(model.state_dict(), FILE)
        images = images.to('cpu')
        outputs = outputs.to('cpu').detach()
        for idx in range(6):
            plt.subplot(2*2, 3, idx+1)
            tmp = images[idx,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp, cmap='gray')

            plt.subplot(2*2, 3, idx+1+6)
            tmp1 = outputs[idx,:,:,:]
            tmp1 = torch.permute(tmp1,(1,2,0))
            plt.imshow(tmp1, cmap='gray')
        plt.show()

        pass

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), FILE)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    loss_in_epoch_loop1 = loss_sum_in_epoch_loop1 / (i + 1)
    loss_in_epoch_loop2 = loss_sum_in_epoch_loop2 / (i + 1)
    print(f'Elapsed time for one epoch: {elapsed_time:.6f} seconds, epoch avg loss={loss_in_epoch_loop1+loss_in_epoch_loop2:.4f}')
    loss_sum_in_epoch_loop1 = 0
    loss_sum_in_epoch_loop2 = 0

