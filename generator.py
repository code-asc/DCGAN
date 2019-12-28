import torch

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.gen_model = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(100, 64*8, 4, 1, 0, bias=False),
                        torch.nn.BatchNorm2d(512),
                        torch.nn.ReLU(),

                        torch.nn.ConvTranspose2d(512, 64*4, 4, 2, 1, bias=False),
                        torch.nn.BatchNorm2d(256),
                        torch.nn.ReLU(),

                        torch.nn.ConvTranspose2d(256, 64*2, 4, 2, 1, bias=False),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU(),

                        torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU(),

                        torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                        torch.nn.Tanh()
                        )

    def forward(self, x):
        return self.gen_model(x)
