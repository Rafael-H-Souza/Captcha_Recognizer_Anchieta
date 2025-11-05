# Reconhecimento Automático de Padrões em Imagens para Validação de Acessos Digitais

Solução completa de Inteligência Artificial e Visão Computacional para decodificar imagens de CAPTCHA. O projeto entrega
um pipeline de Machine Learning otimizado para GPU, desde a criação do dataset até o monitoramento do desempenho do modelo
em produção.

## 📚 Sumário
- [Visão Geral](#-visão-geral)
- [Principais Funcionalidades](#-principais-funcionalidades)
- [Arquitetura do Projeto](#-arquitetura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Configuração do Ambiente](#-configuração-do-ambiente)
- [Aquisição de Dados](#-aquisição-de-dados)
- [Pipeline de Treinamento](#-pipeline-de-treinamento)
- [Monitoramento e Relatórios](#-monitoramento-e-relatórios)
- [Inferência](#-inferência)
- [Estrutura de Pastas](#-estrutura-de-pastas)
- [Variáveis de Ambiente](#-variáveis-de-ambiente)
- [Referências e Fontes de Dados](#-referências-e-fontes-de-dados)

## 🔍 Visão Geral
- **Objetivo:** validar acessos digitais decodificando textos presentes em CAPTCHAs.
- **Modelo:** CNN + GRU, treinado com TensorFlow/Keras.
- **Aceleração:** uso opcional de GPU NVIDIA com crescimento dinâmico de memória.
- **Observabilidade:** logs estruturados, TensorBoard, relatórios automáticos e checkpoints do melhor modelo.

## 🚀 Principais Funcionalidades
- Geração automática de datasets sintéticos de CAPTCHA.
- Pré-processamento de imagens (remoção de ruído, binarização e segmentação).
- Pipeline de treinamento com callbacks de *checkpoint*, *early stopping* e TensorBoard.
- Avaliação automatizada com histórico versionado (`evaluation_log.csv`).
- Relatório diário de progresso com projeção para metas de acurácia e prazos (`make report`).
- Script de inferência pronto para uso em produção.

## 🧱 Arquitetura do Projeto
O código foi dividido em módulos independentes para facilitar manutenção e escalabilidade:
- **`src/config`** – gerenciamento de configurações, leitura de `.env` e setup de GPU.
- **`src/data_loader`** – carregamento e transformação dos dados utilizando `tf.data`.
- **`src/preprocessing`** – filtros e utilitários de preparação de imagens.
- **`src/model`** – arquitetura, treinamento e avaliação do modelo.
- **`src/inference`** – classe de predição com decodificação das saídas.
- **`src/utils`** – logging padronizado e utilidades gerais.
- **`scripts`** – ponto de entrada para geração de dados, preparação, treino, avaliação, relatórios e inferência.

## 🛠️ Pré-requisitos
- Python 3.10+
- GPU NVIDIA com drivers e CUDA/cuDNN (opcional, recomendado)
- [Make](https://www.gnu.org/software/make/)

## ⚙️ Configuração do Ambiente
1. Clone o repositório e acesse a pasta do projeto.
2. Execute o setup completo (cria `venv` e instala dependências):
   ```bash
   make setup
   ```
3. Ative a *venv* manualmente, se precisar rodar scripts diretamente:
   ```bash
   source venv/bin/activate
   ```

## 🗂️ Aquisição de Dados
Você pode usar dados sintéticos ou datasets públicos.

### 1. Gerar CAPTCHAs sintéticos (recomendado)
```bash
make dataset NUM_IMAGES=5000
make preprocess
```
Ou diretamente pelo script:
```bash
source venv/bin/activate
python scripts/generate_dataset.py --num-images 5000 --charset abc123 --length 5
python scripts/prepare_data.py
```

### 2. Baixar datasets públicos
Baixe, descompacte e coloque as imagens em `data/raw/`, criando o arquivo `data/labels/labels.csv` com as colunas
`filename,label`.

| Dataset | Descrição | Link |
| --- | --- | --- |
| CAPTCHA Version 2 | 1 040 imagens de 5 caracteres minúsculos | https://www.kaggle.com/datasets/fournierp/captcha-version-2-images |
| CAPTCHA Dataset | CAPTCHAs com letras e números | https://www.kaggle.com/datasets/imrandude/captcha-dataset |
| Handwritten CAPTCHA | CAPTCHAs simulando escrita manual | https://www.kaggle.com/datasets/shubhammehta21/handwritten-captcha |

Ajuste `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `MAX_LENGTH` e `CHARSET` no arquivo `.env` (ou diretamente em `src/config/settings.py`)
para corresponder ao dataset escolhido.

## 🧠 Pipeline de Treinamento
1. **Preparar dados:** `make preprocess`
2. **Treinar modelo:**
   ```bash
   make train
   ```
   Customize épocas e tamanho do batch diretamente pelo Makefile:
   ```bash
   make train EPOCHS=75 BATCH_SIZE=64
   ```
   Ou chame o script com parâmetros específicos:
   ```bash
   python scripts/train.py --epochs 75 --batch-size 64 --learning-rate 0.0005 --no-gpu
   ```
3. **Avaliação automática:** o treinamento executa `scripts/evaluate.py` ao final, registrando métricas em `logs/evaluation/` e
salvando o melhor checkpoint em `models/checkpoints/best_model.h5`.
4. **Exportação:** o modelo final é salvo em `models/exported/captcha_model.h5`.

## 📊 Monitoramento e Relatórios
- **TensorBoard:**
  ```bash
  source venv/bin/activate
  tensorboard --logdir logs/training
  ```
- **Relatório de progresso:** gera resumo de acurácia atual, percentual da meta e projeção para o prazo definido.
  ```bash
  make report
  make report TARGET_ACCURACY=0.97 DEADLINE=2024-12-31
  python scripts/report.py --target 0.97 --deadline 2024-12-31
  ```
- **Log estruturado:** cada avaliação acrescenta uma linha em `logs/evaluation/evaluation_log.csv`, com timestamp, *loss*,
acurácia média e quantidade de imagens utilizadas.

## 🤖 Inferência
1. Garanta que `models/exported/captcha_model.h5` exista (rodando `make train`).
2. Execute a predição:
   ```bash
   make infer IMAGE_PATH=data/raw/exemplo.png
   ```
   ou
   ```bash
   python scripts/infer.py --image_path caminho/para/imagem.png
   ```

## 🗃️ Estrutura de Pastas
```
pattern_recognition_ai/
├── Makefile
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels/
├── logs/
│   ├── training/
│   └── evaluation/
├── models/
│   ├── checkpoints/
│   └── exported/
├── notebooks/
├── scripts/
└── src/
```

## ⚙️ Variáveis de Ambiente
Copie `.env.example` para `.env` e personalize conforme necessário:
```
DATA_PATH=data/
MODEL_PATH=models/
LOGS_PATH=logs/
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
IMAGE_WIDTH=200
IMAGE_HEIGHT=50
```

## 🔗 Referências e Fontes de Dados
- Documentação TensorFlow: https://www.tensorflow.org/api_docs
- Biblioteca captcha: https://pypi.org/project/captcha/
- Conjuntos de dados recomendados:
  - https://www.kaggle.com/datasets/fournierp/captcha-version-2-images
  - https://www.kaggle.com/datasets/imrandude/captcha-dataset
  - https://www.kaggle.com/datasets/shubhammehta21/handwritten-captcha

---
Com esse fluxo você acompanha a performance de treinamento e aprendizagem em tempo real, identifica gargalos rapidamente e
mantém um histórico completo de experimentos e entregas.
