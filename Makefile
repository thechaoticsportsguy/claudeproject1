.PHONY: install run lint typecheck test clean

install:
	pip install -e ".[dev]"

run:
	streamlit run app/streamlit_app.py

lint:
	ruff check .

typecheck:
	mypy core/ prompts/

test:
	pytest tests/ -v

clean:
	rm -rf outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
