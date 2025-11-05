# Makefile para automação de setup, treino e inferência (Ubuntu/Linux)
# -----------------------------------------------------------------------------

VENV_DIR=venv
PYTHON_BIN=$(VENV_DIR)/bin/python
PIP_BIN=$(VENV_DIR)/bin/pip
EPOCHS ?= 200
BATCH_SIZE ?= 16
IMAGE_PATH ?= data/raw/example_captcha.png
NUM_IMAGES ?= 100
TARGET_ACCURACY ?= 0.95
DEADLINE ?= 2026-12-31

# -----------------------------------------------------------------------------
.PHONY: setup install train train-gpu evaluate infer clean freeze all report dataset preprocess

all: setup freeze

setup:
	@echo "🔧 Criando ambiente virtual em '$(VENV_DIR)'..."
	python3.10 -m venv $(VENV_DIR)
	@echo "✅ Ambiente criado em $(VENV_DIR)."
	@$(MAKE) install

install:
	@echo "📦 Instalando dependências do 'requirements.txt'..."
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt
	@echo "✅ Dependências instaladas com sucesso no venv."

train:
	@echo "🚀 Iniciando treinamento do modelo (CPU)..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/train.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)
	@echo "✅ Treinamento concluído. Executando avaliação pós-treino..."
	$(MAKE) evaluate

train-gpu:
	@echo "🚀 Iniciando treinamento do modelo (GPU)..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/train.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --use-gpu
	@echo "✅ Treinamento concluído. Executando avaliação pós-treino..."
	$(MAKE) evaluate

evaluate:
	@echo "📦 Exportando todos os modelos..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/export_all_models.py
	@echo "📊 Avaliando modelos exportados..."
	@if PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/evaluate.py; then \
		echo "✅ Avaliação concluída com sucesso."; \
	else \
		echo "❌ Erro durante a avaliação. Verifique os logs."; \
	fi

report-data:
	@echo "📄 Gerando relatório de progresso..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/dataset_report.py --target $(TARGET_ACCURACY) --deadline $(DEADLINE)

report:
	@echo "📄 Gerando relatório de progresso..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/report.py --target $(TARGET_ACCURACY) --deadline $(DEADLINE)


infer:
	@echo "🧠 Executando inferência..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/infer.py --image_path $(IMAGE_PATH)
	@echo "✅ Inferência concluída."

dataset:
	@echo "🗂️ Gerando dataset sintético de CAPTCHAs..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/generate_dataset.py --num-images $(NUM_IMAGES)

preprocess:
	@echo "🧼 Pré-processando imagens..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/prepare_data.py

clean:
	@echo "🧹 Limpando cache, logs e checkpoints..."
	rm -rf $(shell find . -name __pycache__)
	rm -rf .pytest_cache/
	rm -rf logs/*
	rm -rf models/checkpoints/*
	@echo "✅ Limpeza finalizada."

freeze:
	@echo "🧊 Exportando dependências atualizadas..."
	$(PIP_BIN) freeze > requirements.txt
	@echo "✅ Arquivo requirements.txt atualizado."
