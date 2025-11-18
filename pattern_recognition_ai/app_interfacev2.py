import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
import logging
from pathlib import Path

from src.inference.predictor import Predictor
from src.preprocessing.filters import preprocess_image


# ======================================================
# 📂 Função: listar modelos .keras
# ======================================================
def listar_modelos(pasta):
    p = Path(pasta)
    if not p.exists():
        return []
    return sorted([f for f in p.iterdir() if f.suffix == ".keras"])


# ======================================================
# 📑 Logs
# ======================================================
def setup_logger():
    logs_dir = Path("logs/streamlit")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("streamlit_ui")
    logger.info("🚀 Streamlit iniciado")
    return logger


logger = setup_logger()


# ======================================================
# 🖥️ Config da Página
# ======================================================
st.set_page_config(
    page_title="Validador de CAPTCHA IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "report a bug": None,
        "get help": None,
        "about": None
    }
)

st.title("🤖 Validador de CAPTCHAs com IA")
st.markdown("### Sistema Inteligente treinado para decodificação de CAPTCHAs.")


# ======================================================
# 🧩 Sidebar — seleção de modelo
# ======================================================
st.sidebar.header("⚙️ Configurações")

pasta_modelos = st.sidebar.selectbox(
    "Pasta de modelos",
    ["models/exported", "models/apurados"],
    index=0
)

modelos = listar_modelos(pasta_modelos)

if not modelos:
    st.sidebar.error("Nenhum arquivo .keras encontrado.")
    st.stop()

modelo_escolhido = st.sidebar.selectbox(
    "Selecione o modelo",
    [m.name for m in modelos]
)

modelo_path = str(Path(pasta_modelos) / modelo_escolhido)

debug_mode = st.sidebar.checkbox("Ativar modo Debug", value=False)


# ======================================================
# Carregar Modelo
# ======================================================
@st.cache_resource
def load_model(path, debug=False):
    logger.info(f"Carregando modelo: {path}")
    return Predictor(path, debug=debug)

predictor = load_model(modelo_path, debug=debug_mode)


# ======================================================
# Tabela estilo screenshot
# ======================================================
def render_prediction_table(details, model_name, pred_text):

    types_list = []
    for d in details:
        attempts = d.get("attempts", [])
        char = attempts[0].get("char") if attempts else "?"
        tipo = "num" if char.isdigit() else "letra" if char.isalpha() else "outro"
        types_list.append(f"{char},{tipo}")

    df = pd.DataFrame([{
        "model": model_name,
        "text": pred_text,
        "types": types_list,
        "ensemble": None,
    }])

    st.markdown("### 🔵 Resultado do Modelo")
    st.dataframe(df, use_container_width=True)


# ======================================================
# Tentativas por caractere
# ======================================================
def show_attempts(details):
    st.markdown("### 🔠 Tentativas por caractere")

    for idx, d in enumerate(details):

        attempts = d.get("attempts", [])
        predicted = d.get("predicted", "?")

        st.markdown(f"#### 📌 Caractere {idx} — Previsto: **{predicted}**")

        rows = []
        for att in attempts[:10]:
            char = att.get("char", "?")
            conf = float(att.get("confidence", 0)) * 100
            tipo = "num" if char.isdigit() else "letra" if char.isalpha() else "outro"

            rows.append({
                "tentativa": f"({char}, {tipo})",
                "confiança (%)": round(conf, 4)
            })

        st.table(pd.DataFrame(rows))
        st.markdown("---")


# ======================================================
# Imagens lado a lado (debug)
# ======================================================
def show_debug_images(debug_steps, cols_per_row=3):

    st.markdown("### 🖼 Pipeline de Pré-processamento")

    rows = []
    current = []

    for step in debug_steps:
        img = step.get("image")
        name = step.get("step")

        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        current.append((name, img))

        if len(current) == cols_per_row:
            rows.append(current)
            current = []

    if current:
        rows.append(current)

    for row in rows:
        cols = st.columns(len(row))
        for col, (name, img) in zip(cols, row):
            col.markdown(f"**{name}**")
            col.image(img, use_column_width=True)


def show_preprocess_preview(uploaded_image):
    """Mostra lado a lado: original → pré-processado"""

    img = Image.open(uploaded_image).convert("L")   # força grayscale
    img_np = np.array(img)

    processed = preprocess_image(img_np)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼 Original")
        st.image(img_np, use_column_width=True, clamp=True)

    with col2:
        st.markdown("### 🎯 Após Pré-processamento")
        st.image(processed, use_column_width=True, clamp=True)

# ======================================================
# 📂 Abas
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Manual",
    "🧪 Dataset Local",
    "🧩 Debug / Modelo",
    "🖼 Pré-processamento Visual"
])



# ======================================================
# 📤 Aba 1 — Upload Manual
# ======================================================
with tab1:

    st.subheader("📤 Enviar imagem de CAPTCHA")

    uploaded_file = st.file_uploader("Selecione uma imagem", type=["png", "jpg", "jpeg"])

    if uploaded_file:

        img = Image.open(uploaded_file)
        st.image(img, width=300)

        tmp = Path("data/tmp")
        tmp.mkdir(parents=True, exist_ok=True)

        img_path = tmp / uploaded_file.name
        img.save(img_path)

        pred_text, details = predictor.predict_with_details(str(img_path))

        st.metric("Predição", pred_text)

        render_prediction_table(details, modelo_escolhido, pred_text)
        show_attempts(details)


# ======================================================
# 🧪 Aba 2 — Dataset Local
# ======================================================
with tab2:

    st.subheader("🧪 Testar com dataset local")

    BASE = Path("imagens/archive/samples")

    if not BASE.exists():
        st.warning("⚠️ Pasta de samples não encontrada.")
    else:
        files = sorted([f for f in os.listdir(BASE) if f.lower().endswith(("png", "jpg", "jpeg"))])

        selected = st.selectbox("Escolha uma imagem", files)
        path = BASE / selected

        img = Image.open(path)
        st.image(img, width=300)

        true_text = os.path.splitext(selected)[0]

        pred_text, details = predictor.predict_with_details(str(path))

        st.metric("Texto Real", true_text)
        st.metric("Predição", pred_text)
        st.metric("Acerto?", "✅ Sim" if pred_text == true_text else "❌ Não")

        render_prediction_table(details, modelo_escolhido, pred_text)
        show_attempts(details)


# ======================================================
# 🧩 Aba 3 — Debug Avançado
# ======================================================
with tab3:

    st.subheader("🧩 Debug completo do modelo selecionado")

    st.info(f"🔍 Modelo carregado: `{modelo_path}`")

    st.markdown("### 📑 Arquitetura (summary)")
    from io import StringIO
    buf = StringIO()
    predictor.model.summary(print_fn=lambda x: buf.write(x + "\n"))
    st.text(buf.getvalue())

    st.markdown("---")

    file = st.file_uploader("Envie uma imagem para debug", type=["png", "jpg", "jpeg"], key="debugger")

    if file:

        tmp = Path("data/tmp")
        tmp.mkdir(exist_ok=True)

        path = tmp / file.name
        with open(path, "wb") as f:
            f.write(file.read())

        result, debug_steps, details = predictor.predict_with_debug(str(path))

        st.metric("Resultado da Predição", result)

        show_debug_images(debug_steps)
        render_prediction_table(details, modelo_escolhido, result)
        show_attempts(details)

# ======================================================
# 🖼 Aba 4 — Pré-processamento Visual
# ======================================================
with tab4:

    st.subheader("🖼 Pré-processamento Visual do CAPTCHA")
    st.markdown("Veja exatamente como o filtro converte a imagem antes da predição.")

    uploaded = st.file_uploader("Envie um CAPTCHA", type=["png", "jpg", "jpeg"], key="preprocess")

    if uploaded:
        show_preprocess_preview(uploaded)
