# Desativar ambiente virtual antigo
deactivate

# Ativar ambiente virtual
source venv/bin/activate


# Garantir que o venv está ativo
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar a biblioteca captcha diretamente
pip install captcha



# Conferir se está instalada
pip list | grep captcha


# Verificar versão do Python
./venv/bin/python --version

# Listar bibliotecas instaladas no venv
./venv/bin/pip list

# Gerar dataset sintético de CAPTCHAs
make dataset

# Pré-processar imagens
make preprocess

# Treinar modelo (CPU)
make train

# Treinar modelo (GPU, se disponível)
make train-gpu

# Avaliar modelo
make evaluate

# Inferência em uma imagem de exemplo
make infer

# Instalar visualizador de imagens (opcional)
sudo apt install feh

# Visualizar imagens do dataset (opcional)
feh data/raw/

# Gerar relatório (opcional)
make report

# Limpar arquivos temporários
make clean

# Congelar dependências
make freeze

# Testar instalação de bibliotecas essenciais
python -c "import tensorflow, cv2, flask; print('✅ Tudo OK!')"

# Rodar interface Streamlit
streamlit run app_interface.py



# Construir a imagem

docker build -t captcha-ai:latest .

# Rodar com GPU (NVIDIA)

docker run --rm --gpus all \
    -v $(pwd)/models_apurados:/app/models_apurados \
    -v $(pwd)/data:/app/data \
    captcha-ai:latest
    
# Rodar com CPU

docker run --rm \
    -v $(pwd)/models_apurados:/app/models_apurados \
    -v $(pwd)/data:/app/data \
    captcha-ai:latest
