"""Pipeline style transfer"""
# pylint: disable=R0913, E1136, E1101
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
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


class StyleTransferParameters:  # pylint: disable=W0102
    """Style Transfer paramereters"""

    def __init__(
        self,
        epochs: int = 1000,
        learning_rate: float = 1e-4,
        optimizer: str = "lbfgs",
        style_layers: List[str] = config.DEFAULT_STYLE_LAYERS,
        content_layers: List[str] = config.DEFAULT_CONTENT_LAYERS,
        weights: Dict[str, float] = config.DEFAULT_STYLE_CONTENT_WEIGHTS,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.weights = weights


class PipelineTransferStyle:
    """Pipeline Transfer Stryle"""

    def __init__(
        self,
        transfer_params: StyleTransferParameters,
        similar_images_csv: Optional[str] = None,
    ) -> None:
        self.model = self.__init_model()

        self.params = transfer_params

        self.content_loss = nn.MSELoss()
        self.style_loss = GramMSELoss()
        if similar_images_csv is not None:
            self.similar_predictor = None
            self.similar_images_csv = pd.read_csv(similar_images_csv)
        else:
            self.similar_predictor = PredictorSimilarCartoon()
            self.similar_images_csv = None

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

    def __get_similar_cartoon(self, content_image_path: str, index: int = 1):
        """Get similar cartoon"""
        if self.similar_images_csv is None:
            logging.info("Predicting similar cartoon")
            if self.similar_predictor is None:
                raise ValueError("Predictor is not defined")
            (
                style_cartoon_features_path,
                _,
            ) = self.similar_predictor.get_most_similar_image(content_image_path)
            style_cartoon_path = style_cartoon_features_path.replace(
                "cartoon_features", "cartoon_frames"
            ).replace("npy", "jpg")
            return Image.open(style_cartoon_path)

        logging.info("Using similar cartoon csv")
        content_image_name = os.path.basename(content_image_path)
        if content_image_name in self.similar_images_csv["name"].unique():
            similar_cartoon_path = self.similar_images_csv.loc[
                self.similar_images_csv["name"] == content_image_name
            ][f"similar_cartoon_{index}"].values[0]
            similar_cartoon_path = similar_cartoon_path.replace("cartoongan/", "")
            return Image.open(similar_cartoon_path)
        raise ValueError("Image not found in csv")

    def cartoonize_image(self, image_path: str) -> Image.Image:
        """Cartoonize an image"""
        content_image = Image.open(image_path)
        style_cartoon = self.__get_similar_cartoon(image_path)

        content_image = preprocess(content_image)
        style_cartoon = preprocess(style_cartoon)

        if torch.cuda.is_available():
            content_image = Variable(content_image.unsqueeze(0).cuda())
            style_cartoon = Variable(style_cartoon.unsqueeze(0).cuda())
        else:
            content_image = Variable(content_image.unsqueeze(0))
            style_cartoon = Variable(style_cartoon.unsqueeze(0))

        style_targets = {
            layer_id: GramMatrix()(A).detach()
            for layer_id, A in self.model(style_cartoon, self.params.style_layers)
        }
        content_targets = {
            layer_id: A.detach()
            for layer_id, A in self.model(content_image, self.params.content_layers)
        }

        generated_image = Variable(content_image.data.clone(), requires_grad=True)

        optimizer = self.__get_optimizer(generated_image)

        for _ in tqdm(range(self.params.epochs)):

            def closure():
                """Closure function for optimizer"""
                optimizer.zero_grad()

                output = self.model(
                    generated_image,
                    self.params.style_layers + self.params.content_layers,
                )
                loss = 0

                for layer_id, layer_output in output:

                    if layer_id in self.params.style_layers:
                        loss += self.params.weights[layer_id] + self.style_loss(
                            layer_output, style_targets[layer_id]
                        )
                    elif layer_id in self.params.content_layers:
                        loss += self.params.weights[layer_id] + self.content_loss(
                            layer_output, content_targets[layer_id]
                        )
                    else:
                        logging.warning(
                            layer_id, layer_output, " not in style or content layers"
                        )

                loss.backward()  # type: ignore
                return loss

            optimizer.step(closure)

        return postprocess(generated_image.data[0].cpu().squeeze())


if __name__ == "__main__":

    style_transfer_params = StyleTransferParameters(
        epochs=1000,
        learning_rate=0.001,
        optimizer="lbfgs",
    )

    pipeline_transfer_style = PipelineTransferStyle(
        style_transfer_params, config.SIMILAR_IMAGES_CSV_PATH
    )

    pipeline_transfer_style.cartoonize_image("data/flickr/Images/667626_18933d713e.jpg")
