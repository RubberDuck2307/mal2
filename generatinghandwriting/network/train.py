import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm

from generatinghandwriting.network.net import Generator, Discriminator

batch_size = 64
num_channels = 1
num_classes = 26
image_size = 28
latent_dim = 128

all_letters = np.load('x_letters.npy')
all_labels = np.load('y_letters.npy')

all_letters = all_letters.astype("float32")
all_letters = torch.Tensor(np.reshape(all_letters, (-1, 28, 28, 1)))


all_labels = torch.Tensor(all_labels).long()

dataset = torch.utils.data.TensorDataset(all_letters, all_labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_Net = Generator(num_classes).to(device)
D_Net = Discriminator(num_classes).to(device)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(D_Net.parameters(), lr=0.5e-4)
optimizerG = optim.Adam(G_Net.parameters(), lr=1e-4)

fixed_noise = torch.randn(num_classes, latent_dim, device=device)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 20
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (images, labels) in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        D_Net.zero_grad()
        # Format batch
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]
        images = images.reshape(batch_size, 1, 28, 28)
        reality_label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = D_Net(images, labels).squeeze()
        # Calculate loss on all-real batch
        errD_real = criterion(output, reality_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, latent_dim, device=device)
        # Generate fake image batch with G
        fake = G_Net(noise, labels)
        reality_label.fill_(fake_label)
        # Classify all fake batch with D
        output = D_Net(fake, labels).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, reality_label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()

        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G_Net.zero_grad()
        reality_label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        fake = G_Net(noise, labels)
        output = D_Net(fake, labels).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, reality_label)
        # Calculate gradients for G
        errG.backward()

        # for name, param in G_Net.named_parameters():
        #     if param.grad is not None:
        #         if torch.all(param.grad.abs() < 1e-6):
        #             print(f"Warning: Vanishing gradients detected in {name}")

        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                output = G_Net(fixed_noise, torch.range(0, 25).to(torch.int64).to(device))
                images = output.reshape(num_classes, 28, 28).cpu()

                fig, axes = plt.subplots(6, 5, figsize=(6, 6))

                for i, ax in enumerate(axes.flat):
                    if (i > 25):
                        break
                    ax.imshow(images[i], cmap="gray")
                    ax.axis("off")

                plt.tight_layout()
                plt.show()

        iters += 1
    torch.save(G_Net.state_dict(), f"generator_{epoch}.pth")

