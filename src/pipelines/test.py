# -*- coding: utf-8 -*-
"""test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PT6MRvQpRIok5EHRcxz6M-g6qghEA4jO
"""

from google.colab import drive
import os
drive.mount("/content/drive")

PROJECT_DIRECTORY = "drive/MyDrive/DeepL"
os.chdir(PROJECT_DIRECTORY)

!ls
!pip install -r requirements.txt

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.cartoon_gan import CartoonGan
from src import config
from src.dataset.dataset_pictures import PicturesDataset
from src.preprocessing.preprocessor import Preprocessor

preprocessor = Preprocessor(size=256)

test_pictures_dataset = PicturesDataset(
    train=False,
    transform=preprocessor.picture_preprocessor(),
    size=50
)

cartoon_gan = CartoonGan(
    nb_channels_1_h_l_disc=32,
    nb_channels_1_h_l_gen=64,
    nb_channels_cartoon=3,
    nb_channels_picture=3,
    nb_resnet_blocks=8,
)

cartoon_gan.load_model(
    generator_path=os.path.join(config.WEIGHTS_FOLDER, "pretrained", "pretrained_gen_11.pkl"),
    discriminator_path=os.path.join(config.WEIGHTS_FOLDER, "pretrained", "pretrained_disc_11.pkl")
)

test_pictures_loader = DataLoader(
    dataset=test_pictures_dataset,
    batch_size=8,
    shuffle=False,
    # drop last incomplete batch
    drop_last=False,
    num_workers=2
)

"""### Save images"""

PRETRAINED_FOLDER = os.path.join("data", "results", "pretrained")

cartoons = cartoon_gan.cartoonize_dataset(test_pictures_loader)

for i, cartoon in enumerate(cartoons):
    np.save(os.path.join(PRETRAINED_FOLDER, f"pretrained_{i}.npy"), cartoon)

"""### Display images"""

import matplotlib.pyplot as plt
from PIL import Image
from torchvision  import transforms

# !pip install matplotlib==3.1.3

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

for cartoon_path in os.listdir(PRETRAINED_FOLDER):
    with open(os.path.join(PRETRAINED_FOLDER, cartoon_path), 'rb') as cartoon_file:
      cartoon = np.load(cartoon_file)
      cartoon = (np.stack([cartoon[0], cartoon[1], cartoon[2]], axis=2) + 1) / 2
      print(np.max(cartoon))
      print(np.min(cartoon))
      plt.imshow(cartoon)
      plt.show()

