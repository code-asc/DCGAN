import torch
from generator import Generator
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

required_samples = 16
device= 'gpu' if torch.cuda.is_available() else 'cpu'
path = '/Users/sandeepchowdaryannabathuni/downloads/generator_model.mdl'

model = torch.load(path, map_location=device)

if isinstance(model, torch.nn.DataParallel):
    model = model.module
else:
    model = model


img_list = []

for i in range(required_samples):
    latent = torch.randn(required_samples, 100, 1,1, device=device)
    fake = model(latent).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=10, normalize=True))

plt.subplot(1,1,1)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
