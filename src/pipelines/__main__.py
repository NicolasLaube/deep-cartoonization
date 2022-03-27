"""Main pipeline for the project."""
from src import dataset, models, preprocessing
from src.pipelines.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline(
        architecture=models.Architecture.GANModular,
        architecture_params=models.ArchitectureParamsModular(),
        cartoons_dataset_parameters=dataset.CartoonsDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
            nb_images=4,
        ),
        pictures_dataset_parameters=dataset.PicturesDatasetParameters(
            new_size=(256, 256),
            crop_mode=preprocessing.CropMode.CROP_CENTER,
            ratio_filter_mode=preprocessing.RatioFilterMode.NO_FILTER,
            nb_images=4,
        ),
        init_models_paths=None,
        training_parameters=models.TrainerParams(batch_size=2),
        pretraining_parameters=models.TrainerParams(batch_size=2),
    )

    pipeline.train(2)
