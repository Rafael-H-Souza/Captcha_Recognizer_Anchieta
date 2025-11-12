import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
import logging
from pathlib import Path

# app_interfacev2.py
from src.inference.predictor import Predictor

# ======================================================
# ⚙️ Caminhos e Configurações de Modelo
# ======================================================
def get_model_path(mode="all_models_unified"):
    """Retorna o caminho do modelo conforme o modo."""
    base = Path("models/exported")
    mapping = {
        "all_models_unified": "all_models_unified.keras",
        "five_char_alphanumeric": "five_char_alphanumeric_model.keras",
        "five_char_numb": "five_char_numb_model.keras",
        "five_char_world": "five_char_world_model.keras",
    }
    return str(base / mapping.get(mode, mapping["all_models_unified"]))

config = {
    "logs_dir": "logs",
    "final_model_path": get_model_path("all_models_unified"),  # modelo inicial
}

# ======================================================
# ⚙️ Configuração de Logs
# ======================================================
LOG_DIR = Path(config["logs_dir"]) / "streamlit"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("streamlit_ui")
logger.info("🚀 Iniciando interface Streamlit de Validação de CAPTCHA")

# ======================================================
# ⚙️ Configuração da Página
# ======================================================
st.set_page_config(
    page_title="Validador de CAPTCHA IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 Validador Automático de CAPTCHA")
st.markdown("### Sistema de Reconhecimento de CAPTCHAs com IA")

# ======================================================
# Sidebar - Configurações
# ======================================================
st.sidebar.header("⚙️ Configurações")
model_option = st.sidebar.selectbox(
    "Selecione o modelo:",
    [   "all_models_unified" ,      
        "five_char_alphanumeric",
        "five_char_numb",
        "five_char_world",
    ],
    index=0
)
debug_mode = st.sidebar.checkbox("Modo Debug", value=False)
show_history = st.sidebar.checkbox("Mostrar Histórico de Predições", value=True)

# Atualiza caminho do modelo
config["final_model_path"] = get_model_path(model_option)

# ======================================================
# 🧠 Carregar modelo (cacheado)
# ======================================================
@st.cache_resource
def load_model(model_path, debug=False):
    logger.info(f"🧠 Carregando modelo: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo não encontrado: {model_path}")
        st.error(f"Modelo não encontrado: {model_path}")
        return None
    predictor = Predictor(model_path, debug=debug)
    logger.info("✅ Modelo carregado com sucesso.")
    return predictor

predictor = load_model(config["final_model_path"], debug=debug_mode)
if predictor is None:
    st.stop()

# ======================================================
# Sessão de Histórico
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# Função utilitária para ajustar detalhes de confiança
# ======================================================
def normalize_confidence(details: list):
    """Multiplica confidences por 100 se forem numéricas."""
    if details:
        df_details = pd.DataFrame(details)
        for col in ["confidence", "confidence_alpha", "confidence_number"]:
            if col in df_details.columns:
                df_details[col] = df_details[col].apply(lambda x: x*100 if isinstance(x, (float, int)) else x)
        return df_details
    return None

# ======================================================
# Abas principais
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Manual",
    "🧪 Dataset Local",
    "🧩 Debug / Ensino",
    "📊 Relatório"
])

# ======================================================
# 📤 ABA 1 - Upload Manual
# ======================================================
with tab1:
    st.subheader("📤 Envio Manual de CAPTCHA")
    uploaded_file = st.file_uploader("Selecione uma imagem:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        logger.info(f"📸 Imagem enviada: {uploaded_file.name}")
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem enviada", use_column_width=True)

        temp_dir = Path("data/tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        image.save(temp_path)

        with st.spinner("🔄 Processando imagem..."):
            decoded_text, details = predictor.predict_with_details(str(temp_path))

        logger.info(f"🧠 Resultado: {decoded_text}")
        st.success(f"✅ Texto decodificado: **{decoded_text}**")

        df_details = normalize_confidence(details)
        if df_details is not None:
            st.markdown("#### 🔠 Detalhes por Caractere")
            st.dataframe(df_details, use_container_width=True)

        st.session_state.history.append({
            "modo": "upload",
            "file": uploaded_file.name,
            "text": decoded_text,
            "details": details,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# ======================================================
# 🧪 ABA 2 - Teste com Dataset Local
# ======================================================
with tab2:
    st.subheader("🧪 Teste com Dataset Local")
    BASE_PATH = Path("imagens/archive/samples")

    if not BASE_PATH.exists():
        st.warning("⚠️ Pasta de imagens não encontrada.")
    else:
        image_files = sorted([f for f in os.listdir(BASE_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        selected_image = st.selectbox("Escolha uma imagem", image_files)
        image_path = BASE_PATH / selected_image
        image = Image.open(image_path)
        st.image(image, caption=f"Imagem: {selected_image}", width=300)

        true_label = os.path.splitext(selected_image)[0]
        predicted_text, details = predictor.predict_with_details(str(image_path))
        confidence = np.mean([d.get("confidence", 0) for d in details]) if details else 0

        logger.info(f"🔍 Dataset: {selected_image} | Previsto: {predicted_text} | Real: {true_label}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Texto Real", true_label)
        col2.metric("Predição", predicted_text)
        col3.metric("Confiança (%)", f"{confidence*100:.2f}")

        st.write(f"**Resultado:** {'✅ Acertou!' if predicted_text.lower()==true_label.lower() else '❌ Errou!'}")

        df_details = normalize_confidence(details)
        if df_details is not None:
            st.markdown("#### 🔠 Detalhes por Caractere")
            st.dataframe(df_details, use_container_width=True)


# ======================================================
# 🧩 ABA 3 - Debug / Ensino
# ======================================================
with tab3:
    st.subheader("🧠 Treinamento Assistido / Debug")
    uploaded_file = st.file_uploader("Envie CAPTCHA (debug)", type=["png", "jpg", "jpeg"], key="debug_upload")

    if uploaded_file:
        temp_dir = Path("data/tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        img_path = temp_dir / uploaded_file.name
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        # usa predict_with_debug
        result, debug_steps, details = predictor.predict_with_debug(str(img_path))
        st.success(f"🔍 Resultado do modelo: **{result}**")

        # Mostrar cada etapa do pré-processamento
        with st.expander("🖼 Histórico de Pré-processamento"):
            for i, step in enumerate(debug_steps):
                # Nome da etapa
                step_name = step.get("step", f"Step {i}") if isinstance(step, dict) else f"Step {i}"
                img = step.get("image") if isinstance(step, dict) else step
                if img is None:
                    continue

                # Garantir que a imagem seja RGB para o Streamlit
                if isinstance(img, np.ndarray):
                    if img.ndim == 2:  # imagem grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:  # RGBA -> RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    elif img.shape[2] == 3:  # BGR -> RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)

                st.image(img, caption=f"{i+1}. {step_name}", use_column_width=True)


        df_details = normalize_confidence(details)
        if df_details is not None:
            st.markdown("#### 🔠 Detalhes por Caractere")
            st.dataframe(df_details, use_container_width=True)


# ======================================================
# 📊 ABA 4 - Relatório de Progresso
# ======================================================
with tab4:
    st.subheader("📊 Relatório de Progresso")
    history_file = Path(config["logs_dir"]) / "evaluation" / "report_history.csv"

    if history_file.exists():
        df = pd.read_csv(history_file, on_bad_lines="skip")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ Nenhum histórico encontrado.")

logger.info("✅ Interface carregada com sucesso.")
