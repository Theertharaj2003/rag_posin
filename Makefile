install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

smoke:
	python -m pytest -q

prepare:
	python run.py --mode demo

retrieve:
	python run.py --mode retrieve --query "What is retrieval augmented generation?"

pipeline:
	python run.py --mode pipeline --query "What is knowledge poisoning in RAG?"

experiment:
	python run.py --mode experiment

ui:
	streamlit run app/main.py
