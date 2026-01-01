from matplotlib import pyplot as plt

from generatinghandwriting.network.net import Generator
import torch

device = "cuda"
G1 = Generator(26).to(device)
G1.load_state_dict(torch.load("generator_19.pth"))
latent_dim = 128

for i in range(26):
    with torch.no_grad():
        output = G1(torch.rand(30, latent_dim).to(device), torch.full((30,), i, dtype=torch.int).to(device))
        images = output.reshape(30, 28, 28).cpu()

        fig, axes = plt.subplots(6, 5, figsize=(6, 6))

        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
