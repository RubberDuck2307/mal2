import torch.nn as nn
import torch
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_classes, 16)
        self.convolution = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), stride=1, in_channels=1, out_channels=16, padding="same"),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(kernel_size=(3, 3), stride=1, in_channels=16, out_channels=32, padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.clazz = nn.Sequential(nn.LazyLinear(out_features=2 * 2 * 32),
                                   nn.LazyBatchNorm1d(),
                                   nn.GELU(),
                                   nn.LazyLinear(out_features=1),
                                   nn.Sigmoid())
        self.flatten = nn.Flatten()
    def forward(self, image, label):
        label = self.embedding(label)
        x = self.convolution(image)
        x = self.flatten(x)
        x = torch.concat((x, label), dim=1)
        x = self.clazz(x)
        return x


class Generator(nn.Module):
    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes, int(num_classes / 2))
        self.fc = nn.Sequential(nn.LazyLinear(out_features=32 * 2 * 2),
                                nn.GELU())
        self.main = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=1),
                                  nn.BatchNorm2d(num_features=32),
                                  nn.GELU(),
                                  nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2),
                                  nn.BatchNorm2d(num_features=16),
                                  nn.GELU(),
                                  nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=6, stride=2),
                                  nn.Sigmoid())

    def forward(self, latent_vector, labels):
        label_embedding = self.embedding(labels)
        x = torch.concat((label_embedding, latent_vector), dim=1)
        x = self.fc(x)
        x = torch.reshape(x, (-1, 32, 2, 2))
        return self.main(x)
