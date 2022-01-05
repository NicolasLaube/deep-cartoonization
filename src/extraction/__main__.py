from src.extraction.train_test_split import (
    create_train_test_frames,
    create_train_test_pictures,
)
from src.extraction.main_csv_extractor import (
    create_all_frames_csv,
    create_all_pictures_csv,
)

create_all_frames_csv()
create_all_pictures_csv()
create_train_test_frames()
create_train_test_pictures()
