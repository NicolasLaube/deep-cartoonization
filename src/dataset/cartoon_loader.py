"""Cartoon dataset Loader"""
from typing import List


from src import config
from src.dataset.image_loader import ImageLoader
from src.dataset.utils import Movie


class CartoonDatasetLoader(ImageLoader):
    """Cartoon dataset loader class"""

    def __init__(
        self, train: bool = True, movies: List[Movie] = config.MOVIES, **kwargs
    ) -> None:
        self.movies = movies
<<<<<<< HEAD
        self.frames = []
        self.__load_frames()

    def __load_frames(self) -> None:
        """Loads the list of frames"""
        for movie in self.movies:
            with open(os.path.join(config.FRAMES_CSV, \
                movie.value + ".csv"), "r", encoding="utf-8") as csv_f:
                reader = csv.reader(csv_f)
                for frame_path in reader:
                    self.frames.append(
                        os.path.join(
                            movie.value, 
                            frame_path[0]
                        )
                    ) 

    def __len__(self) -> int:
        """Length"""
        return len(self.frames)

    def __getitem__(self, index: int) -> NDArray[(Any, Any), np.int32]:
        """Get an item"""
        return cv2.imread(os.path.join(config.FRAMES_FOLDER, self.frames[index]))
=======
        self.train = train
        if train:
            csv_path = config.FRAMES_TRAIN_CSV
        else:
            csv_path = config.FRAMES_TEST_CSV
        ImageLoader.__init__(self, csv_path, **kwargs)
        self._load_specific_frames()

    def _load_specific_frames(self) -> None:
        """Loads the correct list of frames"""
        self.df_images = self.df_images[
            self.df_images["movie"].isin([movie.name for movie in self.movies])
        ]


if __name__ == "__main__":
    loader = CartoonDatasetLoader()
    print(loader[0])
>>>>>>> 1206f5c3419080c7f6dbc860097cc0b7b81ee875
