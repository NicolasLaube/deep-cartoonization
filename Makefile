install:
		pip install -r requirements.txt

install-cpu:
		pip install torch==1.10.1
		pip install torchvision==0.11.2

install-gpu:
		pip install torch==1.9.1+cu102  -f https://download.pytorch.org/whl/cu102/torch_stable.html

install-dev:
		pip install -r requirements.dev

lint:
		python -m pylint src tests
		python -m mypy src tests
		python -m flake8 src tests

test:
		pytest -m "not models and not evaluators"

tests: test
