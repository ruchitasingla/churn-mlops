.PHONY: install data train test lint format run docker-build up down traffic drift-traffic

install:
	pip install -r requirements-dev.txt

data:
	PYTHONPATH=src python -m churn_mlops.data

train:
	PYTHONPATH=src python -m churn_mlops.train

test:
	pytest --cov=churn_mlops --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

run:
	PYTHONPATH=src uvicorn churn_mlops.api:app --reload

docker-build: train
	docker build -t churn-mlops-api:local .

up: train
	docker compose up --build -d

down:
	docker compose down

traffic:
	python scripts/send_traffic.py --drift 0.0 --n 200

drift-traffic:
	python scripts/send_traffic.py --drift 1.0 --n 200
