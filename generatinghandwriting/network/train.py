import os
import shutil
import string

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

from generatinghandwriting.network.net import Generator, Discriminator

import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


batch_size = 64
num_channels = 1
num_classes = 6
image_size = 28
latent_dim = 128
letters = string.ascii_uppercase  # "A" .. "Z"

output_folder = "output"
run_name = "base"
save_path = os.path.join(output_folder, run_name)

if (run_name != "output/test") and os.path.exists(run_name):
    raise ValueError(f"Directory {run_name} already exists!")

if os.path.exists(save_path):
    shutil.rmtree(save_path)

os.makedirs(save_path)


all_letters = np.load('x_letters.npy')
all_labels = np.load('y_letters.npy')

all_letters = all_letters.astype("float32")
all_letters = torch.Tensor(np.reshape(all_letters, (-1, 28, 28, 1)))

all_labels = torch.Tensor(all_labels).long()

letter_mask = all_labels < num_classes
all_letters = all_letters[letter_mask]
all_labels = all_labels[letter_mask]

dataset = torch.utils.data.TensorDataset(all_letters, all_labels)

g = torch.Generator()
g.manual_seed(SEED)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=g
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_Net = Generator(num_classes).to(device)
D_Net = Discriminator(num_classes).to(device)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(D_Net.parameters(), lr=1e-4)
optimizerG = optim.Adam(G_Net.parameters(), lr=1e-4)

fixed_noise = torch.randn(num_classes, latent_dim, device=device)

img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 30
print("Starting Training Loop...")

for epoch in range(num_epochs):

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
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                output = G_Net(fixed_noise, torch.range(0, num_classes - 1).to(torch.int64).to(device))
                images = output.reshape(num_classes, 28, 28).cpu()

                fig, axes = plt.subplots(6, 5, figsize=(6, 6))

                for i, ax in enumerate(axes.flat):
                    if (i == num_classes ):
                        break
                    ax.set_title(letters[i], fontsize=10, pad=2)
                    ax.imshow(images[i], cmap="gray")
                    ax.axis("off")

                plt.tight_layout()
                fig.savefig(f"{save_path}/epoch_{epoch}_iter_{iters}.png")
                plt.close(fig)

        iters += 1
    with open(f"{save_path}/stats", "a") as f:
        f.write('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \n'
          % (epoch, num_epochs, i, len(dataloader),
             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


    torch.save(G_Net.state_dict(), f"generator_{epoch}.pth")
