.PHONY: install pipeline api test clean

install:
	pip install -r requirements.txt

pipeline:
	python run_pipeline.py

api:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest tests/ -v

clean:
	rm -f data/*.parquet models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
