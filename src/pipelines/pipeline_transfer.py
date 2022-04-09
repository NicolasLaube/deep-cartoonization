"""Pipeline style transfer"""
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm

from src import config
from src.models.losses.loss_gram_mse import GramMatrix, GramMSELoss
from src.models.predictor_similar_cartoon import PredictorSimilarCartoon
from src.models.vgg.vgg16_style_transfer import VGGStyleTransfer
from src.preprocessing.transformations.transformations_style_tranfer import (
    postprocess,
    preprocess,
)


@dataclass
class StyleTransferParameters:
    """Style Transfer paramereters"""

    epochs: int = 1000
    learning_rate: float = 1e-4
    optimizer: str = "lbfgs"
    style_layers: List[str] = ["r11", "r21", "r31", "r41", "r51"]
    content_layers: List[str] = ["r42"]
    weights: List[float] = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]] + [1e0]


class PipelineTransferStyle:
    """Pipeline Transfer Stryle"""

    def __init__(self, params: StyleTransferParameters) -> None:
        self.model = self.__init_model()
        self.similar_predictor = PredictorSimilarCartoon()
        self.params = params

        self.content_loss = nn.MSELoss()
        self.style_loss = GramMSELoss()

    @staticmethod
    def __init_model():
        """Initialize model"""
        vgg = VGGStyleTransfer()
        vgg.load_state_dict(torch.load(config.VGG_STYLE_TRANSFERT_WEIGHTS))
        # model won't be trained
        for param in vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            vgg.cuda()
        return vgg

    def __get_optimizer(self, generated_image):
        """Get the optimizer"""
        if self.params.optimizer == "lbfgs":
            optimizer = optim.LBFGS([generated_image.requires_grad_()])
        elif self.params.optimizer == "adam":
            optimizer = optim.Adam(
                [generated_image.requires_grad_()], lr=self.params.learning_rate
            )
        else:
            raise ValueError("Optimizer not supported")
        return optimizer

    def cartoonize_image(self, image_path: str) -> Image.Image:
        """Cartoonize an image"""
        content_image = Image.open(image_path)
        style_cartoon = Image.open(
            self.similar_predictor.get_most_similar_image(content_image)
        )

        content_image = preprocess(content_image)
        style_cartoon = preprocess(style_cartoon)

        if torch.cuda.is_available():
            content_image = Variable(content_image.unsqueeze(0).cuda())
            style_cartoon = Variable(style_cartoon.unsqueeze(0).cuda())
        else:
            content_image = Variable(content_image.unsqueeze(0))
            style_cartoon = Variable(style_cartoon.unsqueeze(0))

        style_targets = [
            GramMatrix()(A).detach()
            for A in self.model(style_cartoon, self.params.style_layers)
        ]
        content_targets = [
            A.detach() for A in self.model(content_image, self.params.content_layers)
        ]

        generated_image = Variable(content_image.data.clone(), requires_grad=True)

        optimizer = self.__get_optimizer(generated_image)

        for _ in tqdm(range(self.params.epochs)):
            optimizer.zero_grad()

            output = self.model(
                generated_image, self.params.style_layers + self.params.content_layers
            )
            loss = 0

            for layer_name, layer_output in enumerate(output):
                if layer_name in self.params.content_layers:
                    loss += self.params.weights[layer_name] + self.content_loss(
                        layer_output, style_targets
                    )
                elif layer_name in self.params.style_layers:
                    loss += self.params.weights[layer_name] + self.style_loss(
                        layer_output, content_targets
                    )

            loss.backward()  # type: ignore
            optimizer.step(loss)

        return postprocess(generated_image.data[0].cpu().squeeze())


if __name__ == "__main__":

    style_transfer_params = StyleTransferParameters(
        epochs=1000,
        learning_rate=0.001,
        optimizer="lbfgs",
    )

    pipeline_transfer_style = PipelineTransferStyle(style_transfer_params)

    pipeline_transfer_style.cartoonize_image(
        "../../../data/images/cartoon/cartoon_1.jpg"
    )
