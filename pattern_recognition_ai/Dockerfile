FROM python:3.10-slim

# Evita buffering de stdout e logs do TensorFlow
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=0

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    git \
    libsm6 libxext6 libxrender-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# -------------------------
# Passo 1: instalar dependências Python
# -------------------------
# Copia apenas requirements.txt para usar cache do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Passo 2: copiar código restante
# -------------------------
COPY . .

# -------------------------
# Entrypoint padrão
# -------------------------
# Abre um bash para poder rodar make dentro do container
CMD ["bash"]
