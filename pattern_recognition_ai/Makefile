# =====================================================================
# 🚀 MAKEFILE - Automação de setup, treino e inferência de CAPTCHAs
# Sistema de Reconhecimento de Padrões com TensorFlow + Streamlit
# =====================================================================

VENV_DIR := venv
PYTHON_BIN := $(VENV_DIR)/bin/python
PIP_BIN := $(VENV_DIR)/bin/pip
STREAMLIT := $(VENV_DIR)/bin/streamlit

EPOCHS := 1
BATCH_SIZE := 32
IMAGE_PATH := data/raw/example_captcha.png
NUM_IMAGES := 10
TARGET_ACCURACY := 0.98
DEADLINE := 2026-12-31

RAW_TYPES := raw single_char five_char
SUBTYPES := numb world alphanumeric alphanumeric_case

PROCESSED_DIR := data/processed
DATASET_DIR := data/dataset
MODELS_DIR := models_apurados
MODEL_PATH := models/exported/captcha_model.h5

# =====================================================================
# 📦 ALVOS DISPONÍVEIS
# =====================================================================
.PHONY: all setup install train train-fast train-gpu evaluate infer clean freeze report dataset decompress-data compress-data preprocess compress-processed decompress-processed inspect-model interface

# =====================================================================
# 🧩 CONFIGURAÇÃO DO AMBIENTE
# =====================================================================
setup:
	@echo "🔧 Criando ambiente virtual em '$(VENV_DIR)' usando Python 3.10..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3.10 -m venv $(VENV_DIR); \
		echo "✅ Ambiente criado em $(VENV_DIR)."; \
	else \
		echo "⚡ Ambiente virtual já existe."; \
	fi
	@echo "📦 Instalando dependências..."
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt
	@echo "🗂️ Criando estrutura de diretórios..."
	@mkdir -p $(PROCESSED_DIR) $(DATASET_DIR) $(MODELS_DIR)
	@for t in $(RAW_TYPES); do \
		for st in $(SUBTYPES); do \
			mkdir -p $(PROCESSED_DIR)/$$t/$$st; \
			mkdir -p $(DATASET_DIR)/$$t/$$st; \
		done; \
	done
	@echo "✅ Setup concluído com sucesso!"

# =====================================================================
# 🧠 TREINAMENTO
# =====================================================================
train:
	@echo "🚀 Treinamento do modelo (CPU)..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/train.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)
	@echo "✅ Treinamento concluído!"
	$(MAKE) evaluate

train-fast:
	@echo "🚀 Treinamento rápido do modelo..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/train_fast.py --dataset data/dataset/five_char_alphanumeric_case.npz --epochs 20 --batch_size 64 --model_out $(MODELS_DIR)/five_char_alphanumeric_case_model.h5
	@echo "✅ Treinamento rápido concluído!"

train-gpu:
	@echo "🚀 Treinamento do modelo (GPU)..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/train.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --use-gpu
	@echo "✅ Treinamento concluído!"
	$(MAKE) evaluate

# =====================================================================
# 📊 AVALIAÇÃO E RELATÓRIO
# =====================================================================
evaluate:
	@echo "📊 Avaliando modelo treinado..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/evaluate.py
	@echo "✅ Avaliação concluída."

report:
	@echo "📄 Gerando relatório de progresso..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/report.py --target $(TARGET_ACCURACY) --deadline $(DEADLINE)

# =====================================================================
# 🧠 INFERÊNCIA E TESTES
# =====================================================================
infer:
	@echo "🧠 Executando inferência..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/infer.py --image_path $(IMAGE_PATH)
	@echo "✅ Inferência concluída."

# =====================================================================
# 🧩 GERAÇÃO DE DADOS
# =====================================================================
dataset:
	@echo "🗂️ Gerando dataset sintético de CAPTCHAs..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/generate_dataset.py --num-images $(NUM_IMAGES)


# =====================================================================
# 🗜️ COMPACTAÇÃO / DESCOMPACTAÇÃO DA PASTA DATA
# =====================================================================
decompress-data:
	@echo "📦 Descompactando a pasta data..."
	@if [ -f data/data.zip ]; then \
		unzip -o data/data.zip -d ./; \
		echo "✅ Pasta data descompactada."; \
	else \
		echo "⚡ Nenhum arquivo zip encontrado em data/data.zip."; \
	fi

compress-data:
	@echo "🗜️ Compactando a pasta data..."
	@zip -r data/data.zip data/*
	@echo "✅ Pasta data compactada."

# =====================================================================
# 🧼 PRÉ-PROCESSAMENTO
# =====================================================================
preprocess: decompress-data
	@echo "🧼 Pré-processando imagens..."
	PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/prepare_data.py
	$(MAKE) compress-processed

# =====================================================================
# 🗜️ COMPACTAÇÃO / DESCOMPACTAÇÃO
# =====================================================================
decompress-processed:
	@echo "📦 Descompactando dados pré-processados..."
	@if [ -f $(PROCESSED_DIR)/processed_data.zip ]; then \
		unzip -o $(PROCESSED_DIR)/processed_data.zip -d $(PROCESSED_DIR)/; \
		echo "✅ Dados descompactados."; \
	else \
		echo "⚡ Nenhum arquivo zip encontrado."; \
	fi

compress-processed:
	@echo "🗜️ Compactando dados pré-processados..."
	@zip -r $(PROCESSED_DIR)/processed_data.zip $(PROCESSED_DIR)/*
	@echo "✅ Dados compactados."

# =====================================================================
# 🔍 INSPEÇÃO DO MODELO
# =====================================================================
inspect-model: 
	@echo "🔍 Inspecionando modelo: $(MODEL_PATH)"
	@if [ -f "$(MODEL_PATH)" ]; then \
		PYTHONPATH=$(CURDIR) $(PYTHON_BIN) scripts/inspect_model.py --model $(MODEL_PATH); \
	else \
		echo "❌ Modelo não encontrado em $(MODEL_PATH)"; \
		exit 1; \
	fi

# =====================================================================
# 💻 INTERFACE STREAMLIT
# =====================================================================
interface:
	@echo "💻 Iniciando interface Streamlit..."
	PYTHONPATH=$(CURDIR) $(STREAMLIT) run app_interfacev2.py

# =====================================================================
# 🧹 LIMPEZA
# =====================================================================
clean:
	@echo "🧹 Limpando cache, logs e checkpoints..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache/ logs/* models/checkpoints/*
	@echo "✅ Limpeza concluída."

# =====================================================================
# 🧊 EXPORTAR DEPENDÊNCIAS
# =====================================================================
freeze:
	@echo "🧊 Exportando dependências atualizadas..."
	$(PIP_BIN) freeze > requirements.txt
	@echo "✅ Arquivo requirements.txt atualizado."

# =====================================================================
# 🆘 AJUDA
# =====================================================================
help:
	@echo ""
	@echo "🧭 Comandos disponíveis:"
	@echo "  make setup           -> Cria ambiente virtual e estrutura de pastas"
	@echo "  make train           -> Treina o modelo (CPU)"
	@echo "  make train-gpu       -> Treina o modelo (GPU)"
	@echo "  make train-fast      -> Treina modelo rápido (subset)"
	@echo "  make evaluate        -> Avalia o modelo treinado"
	@echo "  make report          -> Gera relatório de progresso"
	@echo "  make infer           -> Executa inferência"
	@echo "  make dataset         -> Gera dataset de treinamento"
	@echo "  make preprocess      -> Pré-processa as imagens (auto compacta/descompacta)"
	@echo "  make compress-processed   -> Compacta dados pré-processados"
	@echo "  make decompress-processed -> Descompacta dados pré-processados"
	@echo "  make inspect-model   -> Mostra detalhes do modelo .h5"
	@echo "  make interface       -> Executa a interface Streamlit"
	@echo "  make clean           -> Limpa caches e logs"
	@echo "  make freeze          -> Atualiza requirements.txt"
	@echo ""
