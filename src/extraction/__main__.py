from src.extraction.train_test_split import (
    create_train_test_cartoons,
    create_train_test_pictures,
)
from src.extraction.main_csv_extractor import (
    create_all_cartoons_csv,
    create_all_pictures_csv,
)

create_all_cartoons_csv()
create_all_pictures_csv()
create_train_test_cartoons()
create_train_test_pictures()
