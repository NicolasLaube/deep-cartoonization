from src import models, preprocessing

architecture_dict = {
    "Fixed GAN": models.Architecture.GANFixed,
    "Modular GAN": models.Architecture.GANModular,
    "Anime GAN": models.Architecture.GANAnime,
}

crop_mode_dict = {
    "RESIZE": preprocessing.CropMode.RESIZE,
    "CROP_CENTER": preprocessing.CropMode.CROP_CENTER,
    "CROP_RANDOM": preprocessing.CropMode.CROP_RANDOM,
}

ratio_filter_dict = {
    "NO_FILTER": preprocessing.RatioFilterMode.NO_FILTER,
    "FILTER_PORTRAIT": preprocessing.RatioFilterMode.FILTER_PORTRAIT,
    "FILTER_LANDSCAPE": preprocessing.RatioFilterMode.FILTER_LANDSCAPE,
}
