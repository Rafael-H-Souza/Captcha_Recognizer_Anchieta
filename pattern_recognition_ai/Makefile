# Makefile para automação de setup, treino e inferência (Ubuntu/Linux)
# -----------------------------------------------------------------------------

VENV_DIR=venv
PYTHON_BIN := python3
PYTHON_BIN=$(VENV_DIR)/bin/python
PIP_BIN=$(VENV_DIR)/bin/pip
# Número de épocas de treinamento do modelo
EPOCHS ?= 120                
# Tamanho do batch de imagens por iteração
BATCH_SIZE ?= 32               
# Quantidade total de imagens no dataset
NUM_IMAGES ?= 50000           
# Acurácia alvo para considerar o treino satisfatório
TARGET_ACCURACY ?= 0.95        
# Prazo final para finalizar o treino ou avaliação
DEADLINE ?= 2025-11-14       
# Caminho da imagem de teste ou exemplo
IMAGE_PATH ?= five_char_alphanumeric_case_1.png 


# -----------------------------------------------------------------------------
.PHONY: setup install train train-gpu evaluate infer clean freeze all report dataset preprocess report-data interface merge-models show-model generate-images

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

train-parallel:
	@echo "⚡ Iniciando TREINO PARALELO (CPU Turbo)..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/train.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) > logs/train_parallel.log 2>&1 &
	@echo "🔥 Treino paralelo rodando em background! (logs/train_parallel.log)"
	
train-parallel-live:
	@echo "⚡ Iniciando treino paralelo (LIVE)..."
	(PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/train.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)) &


show-model:
	@echo "🚀 Iniciando treinamento do modelo (CPU)..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/show_models.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)
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

generate-images:
	@echo "🖼️ Gerando imagens pré-processadas em debug/processed_imgs/..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/generate_preprocessed_images.py


preprocess:
	@echo "🧼 Pré-processando imagens..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/prepare_data.py

interface:
	@echo "🧠 Executando interface..."
	. venv/bin/activate && streamlit run app_interfacev2.py
	@echo "✅ Inferência concluída."

merge-models:
	@echo "🔀 Mesclando modelos treinados..."
	venv/bin/python merge_models.py
	@echo "✅ Modelos mesclados com sucesso."

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
