"""Cartoon classifier"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models, transforms
from tqdm import tqdm

from src import config
from src.dataset.dataset_classification import CustomImageDataset


class CartoonClassifier:
    """Cartoon Classifier"""

    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.__init_model()

    @staticmethod
    def __init_model():
        """Initialize the model"""
        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        nb_features = model.fc.in_features

        model.fc = nn.Linear(nb_features, 2)
        return model

    def load_model(self, model_path: str = config.CLASSIFIER_WEIGHTS):
        """Load the model"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        summary(self.model, (3, 224, 224))

    @staticmethod
    def preprocess_train():
        """Preprocess images"""

        return transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(224),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def preprocess_validation():
        """Preprocess images"""

        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def train(self, df_train, df_validation):  # pylint: disable=R0914
        """Train the classifier"""
        train_data = CustomImageDataset(df_train, self.preprocess_train)
        val_data = CustomImageDataset(df_validation, self.preprocess_validation)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        pretrained_criterion = nn.CrossEntropyLoss()
        pretrained_optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=0.001)

        for epoch in tqdm(range(10)):
            for phase in ["train", "val"]:

                running_loss = 0.0
                running_corrects = 0

                if phase == "train":
                    self.model.train()  # Set model to training mode
                    dataset = train_loader
                else:
                    self.model.eval()  # Set model to evaluate mode
                    dataset = valid_loader

                for cartoon_or_image, label in tqdm(dataset):
                    cartoon_or_image = cartoon_or_image.to(self.device)

                    self.model.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        output = self.model(cartoon_or_image)
                        loss = pretrained_criterion(output, label)
                    if phase == "train":
                        loss.backward()
                        pretrained_optimizer.step()

                    running_loss += loss.item() * cartoon_or_image.size(0)
                    running_corrects += label == output

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)

            print(f"{phase} loss: {epoch_loss:.4f}")
            print(f"{phase} acc: {epoch_acc:.4f}")

            self.save_model(f"classifier_weights_epoch_{epoch}.pt")

    def test(self, df_test):
        """Test the classifier"""
        test_data = CustomImageDataset(df_test, self.preprocess_validation)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        self.model.eval()

        true_positif = 0
        true_negatif = 0
        false_positif = 0
        false_negatif = 0
        for cartoon_or_image, labels in tqdm(test_loader):
            cartoon_or_image = cartoon_or_image.to(self.device)

            output = self.model(cartoon_or_image)
            _, preds = torch.max(output, 1)

            for pred, label in zip(preds, labels):

                if pred.item() == label.item():
                    if pred.item() == 0:
                        true_negatif += 1
                    else:
                        true_positif += 1
                else:
                    if pred.item() == 0:
                        false_negatif += 1
                    else:
                        false_positif += 1

        print(f"True positive: {true_positif}")
        print(f"True negative: {true_negatif}")
        print(f"False positive: {false_positif}")
        print(f"False negative: {false_negatif}")

        print(
            f"""Accuracy: {(true_positif + true_negatif) /
            (true_positif + true_negatif + false_positif + false_negatif)}"""
        )
        print(f"Precision: {true_positif / (true_positif + false_positif)}")
        print(f"Recall: {true_positif / (true_positif + false_negatif)}")
        print(
            f"F1 score: {2 * true_positif / (2 * true_positif + false_positif + false_negatif)}"
        )

    def save_model(self, model_path: str) -> None:
        """Save the model"""
        torch.save(self.model, model_path)

    def predict_from_path(self, image_path: str):
        """Predict the class of an image"""
        if ".npy" in image_path:
            image = np.load(image_path)
            image = Image.fromarray(image)
        else:

            image = Image.open(image_path)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        image = self.preprocess_validation()(image)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            _, preds = torch.max(output, 1)
        return preds.item()

    def evaluate_from_folder(self, folder_path: str):
        """Evaluates from folder"""
        files = os.listdir(folder_path)
        files = [os.path.join(folder_path, file) for file in files]  # if "_50" in file

        true_positif = 0
        false_negatif = 0

        for file in tqdm(files):
            pred = self.predict_from_path(file)
            if pred == 0:
                false_negatif += 1
            else:
                true_positif += 1

        print(f"True positive: {true_positif}")
        print(f"False negative: {false_negatif}")

        print(
            f"""Accuracy: {(true_positif) /
            (true_positif + false_negatif)}"""
        )
        print(f"Precision: {true_positif / (true_positif)}")
        print(f"Recall: {true_positif / (true_positif + false_negatif)}")
        print(f"F1 score: {2 * true_positif / (2 * true_positif + false_negatif)}")


if __name__ == "__main__":
    import os

    from sklearn.model_selection import train_test_split

    # ONLY IMAGES IN THE "CARTOONIZATION TRAIN DATASET" ARE USED FOR TRAINING AND TESTING

    IMAGE_DF = pd.read_csv("data/pictures_train.csv")
    IMAGE_DF["label"] = 0
    CARTOONS_DF = pd.read_csv("data/cartoons_train.csv")
    CARTOONS_DF["label"] = 1

    ALL_DATA = pd.concat([IMAGE_DF, CARTOONS_DF], axis=0)

    ALL_DATA = ALL_DATA.groupby("label")
    ALL_DATA = pd.DataFrame(
        ALL_DATA.apply(
            lambda x: ALL_DATA.sample(
                ALL_DATA.size().min(), random_state=config.RANDOM_STATE
            ).reset_index(drop=True)
        )
    )
    ALL_DATA = ALL_DATA.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(
        drop=True
    )

    DF_TRAIN_VAL, DF_TEST = train_test_split(
        ALL_DATA, test_size=0.2, random_state=config.RANDOM_STATE
    )
    DF_TRAIN, DF_VAL = train_test_split(
        DF_TRAIN_VAL, test_size=0.2, random_state=config.RANDOM_STATE
    )

    CARTOON_CLASSIFIER = CartoonClassifier()
    CARTOON_CLASSIFIER.load_model()

    # CARTOON_CLASSIFIER.train(DF_TRAIN, DF_VAL)
    # CARTOON_CLASSIFIER.test(DF_TEST)
    # print(
    #     CARTOON_CLASSIFIER.predict_from_path("data/flickr/Images/667626_18933d713e.jpg")
    # )
    # print(
    #     CARTOON_CLASSIFIER.predict_from_path(
    #         "data/cartoon_frames/BabyBoss/frame0-01-04.40.jpg"
    #     )
    # )

    CARTOON_CLASSIFIER.evaluate_from_folder(
        "data/results/lr_1_epoch_10_optimizer_lbfgs"
    )
