import torch

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis_model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),

                torch.nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(64 * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),

                torch.nn.Conv2d(64 * 2, 64 * 2 * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(64 * 2 * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),

                torch.nn.Conv2d(64 * 2 * 2, 64 * 2 * 2 * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(64 * 2 * 2 * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),

                torch.nn.Conv2d(64 * 2 * 2 * 2, 1, 4, 1, 0, bias=False),
                torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis_model(x)
