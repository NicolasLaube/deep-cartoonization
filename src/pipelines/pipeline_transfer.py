"""Pipeline style transfer"""
# pylint: disable=R0913, E1136, E1101
import csv
import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
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


class StyleTransferParameters:  # pylint: disable=W0102, R0902
    """Style Transfer paramereters"""

    def __init__(
        self,
        save_path: str,
        epochs: int = 1000,
        learning_rate: float = 1e-4,
        optimizer: str = "lbfgs",
        style_layers: List[str] = config.DEFAULT_STYLE_LAYERS,
        content_layers: List[str] = config.DEFAULT_CONTENT_LAYERS,
        weights: Dict[str, float] = config.DEFAULT_STYLE_CONTENT_WEIGHTS,
        num_similar_cartoons: int = config.DEFAULT_NUM_SIMILAR_CARTOONS,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.weights = weights
        self.save_path = save_path
        self.num_similar_cartoons = num_similar_cartoons


class PipelineTransferStyle:
    """Pipeline Transfer Stryle"""

    def __init__(
        self,
        transfer_params: StyleTransferParameters,
        similar_images_csv: Optional[str] = None,
    ) -> None:
        self.model = self.__init_model()

        self.params = transfer_params
        if torch.cuda.is_available():
            self.content_loss = nn.MSELoss().cuda()
            self.style_loss = GramMSELoss().cuda()
        else:
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
            optimizer = optim.LBFGS(
                [generated_image.requires_grad_()], lr=self.params.learning_rate
            )
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
                score,
            ) = self.similar_predictor.get_most_similar_image(content_image_path)
            print(f"Most similar cartoon: {style_cartoon_features_path}")
            print(f"Similarity score: {score}")
            style_cartoon_path = style_cartoon_features_path.replace(
                "cartoon_features", "cartoon_frames"
            ).replace("npy", "jpg")
            return Image.open(style_cartoon_path), score

        logging.info("Using similar cartoon csv")
        content_image_name = os.path.basename(content_image_path)
        if content_image_name in self.similar_images_csv["name"].unique():
            similar_cartoon_path = self.similar_images_csv.loc[
                self.similar_images_csv["name"] == content_image_name
            ][f"similar_cartoon_{index}"].values[0]
            similar_cartoon_path = similar_cartoon_path.replace("cartoongan/", "")
            return Image.open(similar_cartoon_path), None
        raise ValueError("Image not found in csv")

    def save_and_plot(
        self,
        image_path,
        local_gen,
        epoch,
        total_loss_history,
        content_loss_history,
        style_loss_history,
        similarity,
    ):
        """Save and plot"""
        plt.imshow(local_gen)
        plt.show()
        gen_save_path = os.path.join(
            self.params.save_path,
            os.path.basename(image_path.replace(".jpg", f"_{epoch}.npy")),
        )
        np.save(gen_save_path, np.asarray(local_gen))

        loss_file_path = os.path.join(self.params.save_path, "loss.csv")
        if not os.path.exists(loss_file_path):
            with open(loss_file_path, "w", encoding="utf-8") as loss_file:
                writer = csv.writer(loss_file)
                writer.writerow(
                    [
                        "image",
                        "epoch",
                        "total_loss",
                        "content_loss",
                        "style_loss",
                        "similarity",
                    ]
                )
        with open(loss_file_path, "a", encoding="utf-8") as loss_file:
            writer = csv.writer(loss_file)
            if len(total_loss_history) > 0:
                writer.writerow(
                    [
                        image_path,
                        epoch,
                        total_loss_history[-1],
                        content_loss_history[-1],
                        style_loss_history[-1],
                        similarity,
                    ]
                )

    def cartoonize_image(  # pylint: disable=R0915
        self, image_path: str, verbose: bool = False
    ) -> Image.Image:
        """Cartoonize an image"""
        content_image = Image.open(image_path)
        style_cartoon, similarity = self.__get_similar_cartoon(image_path)

        if verbose:
            print("Content Image:")
            plt.imshow(content_image)
            plt.show()
            print("Style Image:")
            plt.imshow(style_cartoon)
            plt.show()

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

        if torch.cuda.is_available():
            style_targets = {
                layer_id: A.cuda() for layer_id, A in style_targets.items()
            }
            content_targets = {
                layer_id: A.cuda() for layer_id, A in content_targets.items()
            }

        generated_image = Variable(content_image.data.clone(), requires_grad=True)

        optimizer = self.__get_optimizer(generated_image)

        total_loss_history = []
        content_loss_history = []
        style_loss_history = []

        for epoch in tqdm(range(self.params.epochs)):

            def closure():
                """Closure function for optimizer"""
                optimizer.zero_grad()

                output = self.model(
                    generated_image,
                    self.params.style_layers + self.params.content_layers,
                )
                loss = 0

                style_loss_values = 0
                content_loss_values = 0

                for layer_id, layer_output in output:

                    if layer_id in self.params.style_layers:
                        style_loss_value = self.params.weights[
                            layer_id
                        ] * self.style_loss(layer_output, style_targets[layer_id])
                        loss += style_loss_value
                        style_loss_values += style_loss_value.item()

                    elif layer_id in self.params.content_layers:
                        content_loss_value = self.params.weights[
                            layer_id
                        ] * self.content_loss(layer_output, content_targets[layer_id])
                        loss += content_loss_value

                        content_loss_values += content_loss_value.item()
                    else:
                        logging.warning(
                            layer_id, layer_output, " not in style or content layers"
                        )

                total_loss_history.append(loss.item())
                content_loss_history.append(content_loss_values)
                style_loss_history.append(style_loss_values)

                loss.backward()  # type: ignore
                return loss

            if epoch % 10 == 0 and verbose and epoch != 0:
                local_gen = postprocess(generated_image.data[0].cpu().squeeze())
                self.save_and_plot(
                    image_path,
                    local_gen,
                    epoch,
                    total_loss_history,
                    content_loss_history,
                    style_loss_history,
                    similarity,
                )

            optimizer.step(closure)

        if verbose:

            self.plot_loss(total_loss_history, "Total Loss")
            print("Total Loss:", total_loss_history[-1])
            self.plot_loss(content_loss_history, "Content Loss")
            print("Content Loss:", content_loss_history[-1])
            self.plot_loss(style_loss_history, "Style Loss")
            print("Style Loss:", style_loss_history[-1])

        return postprocess(generated_image.data[0].cpu().squeeze())

    def plot_loss(self, loss_history: List[float], title: str = ""):
        """Plot loss"""
        plt.plot(loss_history)
        plt.title(title)
        plt.show()
        plt.savefig(os.path.join(self.params.save_path, f"{title}.png"))


if __name__ == "__main__":

    style_transfer_params = StyleTransferParameters(
        epochs=1000,
        learning_rate=0.001,
        optimizer="lbfgs",
        save_path="results/",
    )

    pipeline_transfer_style = PipelineTransferStyle(
        style_transfer_params, config.SIMILAR_IMAGES_CSV_PATH
    )

    pipeline_transfer_style.cartoonize_image("data/flickr/Images/667626_18933d713e.jpg")
