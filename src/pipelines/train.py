import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from src.models.parameters import CartoonGanParameters
from src.dataset.dataset_cartoon import CartoonDataset
from src.dataset.dataset_pictures import PicturesDataset
from src.preprocessing.preprocessor import Preprocessor
from src.models.cartoon_gan import CartoonGan


cuda = torch.cuda.is_available()
print(cuda)

if cuda:
  print(torch.cuda.get_device_name(0))

device = "cuda" if cuda else "cpu"



BATCH_SIZE = 16

pretraining_parameters = CartoonGanParameters(
    epochs=10,
    gen_lr=1e-3,
    disc_lr=1e-2,
    nb_resnet_blocks=3,
    batch_size=BATCH_SIZE,
    conditional_lambda=2
)

training_parameters = CartoonGanParameters(
    epochs=10,
    gen_lr=1e-3,
    disc_lr=1e-2,
    nb_resnet_blocks=3,
    batch_size=BATCH_SIZE,
    conditional_lambda=2
)




preprocessor = Preprocessor(size=256)

pictures_dataset = PicturesDataset()

cartoons_dataset = CartoonDataset(
    train=True,
)


train_pictures_loader = DataLoader(
    dataset=pictures_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # drop last incomplete batch
    drop_last=True,
    num_workers=2
)

train_cartoons_loader = DataLoader(
    dataset=cartoons_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2
)

cartoon_gan = CartoonGan()

cartoon_gan.pretrain(
    pictures_loader=train_pictures_loader,
    parameters=pretraining_parameters
)