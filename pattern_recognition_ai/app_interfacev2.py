import os
import re
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from src.config.settings import config
from src.inference.predictor import Predictor

# ======================================================
# Configuração da página
# ======================================================
st.set_page_config(
    page_title="Validador CAPTCHA IA",
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
debug_mode = st.sidebar.checkbox("Modo Debug", value=False, key="debug_mode")
show_history = st.sidebar.checkbox("Mostrar Histórico", value=True, key="show_history")

# ======================================================
# 🧠 Carregar modelo (cacheado)
# ======================================================
@st.cache_resource
def load_predictor(debug=False):
    """
    Carrega o modelo automaticamente, sem mostrar ensemble.
    """
    try:
        #predictor = Predictor(model_path=None, debug=debug, compile_models=False)
        predictor = Predictor(model_path=None, debug=debug)
        st.success("✅ Modelo carregado com sucesso!")
        return predictor
    except Exception as e:
        st.error(f"⚠️ Não foi possível carregar o modelo: {e}")
        return None

predictor = load_predictor(debug=debug_mode)

if predictor is None:
    st.warning("⚠️ Modelo não carregado. Predições não estarão disponíveis.")

# ======================================================
# Inicializa histórico
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

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
# ABA 1 - Upload Manual
# ======================================================
with tab1:
    st.subheader("📤 Envio Manual de CAPTCHA")
    uploaded_file = st.file_uploader(
        "Selecione uma imagem:", type=["png", "jpg", "jpeg"], key="upload_manual_tab1"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem enviada", width='stretch')

        if predictor:
            temp_dir = "data/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            image.save(temp_path)

            with st.spinner("🔄 Processando imagem..."):
                decoded_text, details = predictor.predict_with_details(temp_path)

            # Calcula confiança média
            confidence = np.mean([d.get("confidence", 0) for d in details]) if details else 0

            st.success(f"✅ Texto decodificado: **{decoded_text}**")
            st.metric("Confiança Média (%)", f"{confidence*100:.2f}")

            if details:
                df_details = pd.DataFrame(details)
                st.markdown("#### 🔠 Detalhes por Caractere")
                st.dataframe(df_details, width='stretch')

            st.session_state.history.append({
                "modo": "upload",
                "file": uploaded_file.name,
                "text": decoded_text,
                "details": details
            })
        else:
            st.info("🔹 Upload apenas visualizado, sem predição disponível.")

# ======================================================
# ABA 2 - Dataset Local
# ======================================================
with tab2:
    st.subheader("🧪 Teste com Dataset Local")
    BASE_PATH = "data/samples/"
    if not os.path.exists(BASE_PATH):
        st.warning("⚠️ Pasta de imagens não encontrada.")
    else:
        image_files = sorted([
            f for f in os.listdir(BASE_PATH)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if image_files:
            selected_image = st.selectbox("Escolha uma imagem", image_files, key="dataset_select_tab2")
            image_path = os.path.join(BASE_PATH, selected_image)
            image = Image.open(image_path)
            st.image(image, caption=f"Imagem: {selected_image}", width=300)

            true_label = os.path.splitext(selected_image)[0]

            if predictor:
                predicted_text, details = predictor.predict_with_details(image_path)
                confidence = np.mean([d.get("confidence", 0) for d in details]) if details else 0

                # Calcula acurácia individual
                acuracia = 1.0 if predicted_text.lower() == true_label.lower() else 0.0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Texto Real", true_label)
                col2.metric("Predição", predicted_text)
                col3.metric("Confiança Média (%)", f"{confidence*100:.2f}")
                col4.metric("Acurácia", "✅" if acuracia == 1 else "❌")

                if details:
                    df_details = pd.DataFrame(details)
                    st.markdown("#### 🔠 Detalhes por Caractere")
                    st.dataframe(df_details, width='stretch')

                # Atualiza histórico para acurácia média
                st.session_state.history.append({
                    "modo": "dataset",
                    "file": selected_image,
                    "text": predicted_text,
                    "true_label": true_label,
                    "details": details,
                    "acuracia": acuracia,
                    "confidence": confidence
                })
            else:
                st.info("🔹 Modelo não carregado. Apenas preview da imagem.")

# ======================================================
# ABA 3 - Debug / Ensino
# ======================================================
with tab3:
    st.subheader("🧠 Treinamento Assistido / Debug")
    temp_dir = "data/tmp/"
    os.makedirs(temp_dir, exist_ok=True)

    predictor_debug = Predictor(debug=True)

    uploaded_file = st.file_uploader(
        "📤 Faça upload da imagem CAPTCHA:", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        safe_name = re.sub(r'[^\w\-_\. ]', '_', uploaded_file.name)
        img_path = os.path.abspath(os.path.join(temp_dir, safe_name))

        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        if os.path.exists(img_path):
            try:
                result, debug_steps, details = predictor_debug.predict_with_debug(img_path)
                st.success(f"🔍 Resultado do modelo: **{result}**")

                with st.expander("🖼 Histórico de Processamento"):
                    cols = st.columns(2)
                    step_titles = {
                        "original": "Imagem Original",
                        "gray": "Convertida para Tons de Cinza",
                        "clahe": "Contraste Ajustado (CLAHE)",
                        "blur": "Aplicado Blur/Gaussian",
                        "thresh": "Binária / Limiarização",
                    }

                    for i, (step_name, img) in enumerate(debug_steps.items()):
                        if isinstance(img, np.ndarray):
                            if img.dtype in [np.float32, np.float64]:
                                img = (img * 255).astype(np.uint8)
                            if len(img.shape) == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            img = Image.fromarray(img)
                        elif not isinstance(img, Image.Image):
                            continue

                        caption = step_titles.get(step_name, step_name)
                        cols[i % 2].image(img, caption=caption, width=450)

                with st.expander("📄 Detalhes da predição"):
                    st.json(details)

            except Exception as e:
                st.error(f"❌ Erro na predição: {e}")
        else:
            st.error(f"❌ Falha ao salvar a imagem em {img_path}")

# ======================================================
# ABA 4 - Relatório
# ======================================================
with tab4:
    st.subheader("📊 Relatório de Progresso")
    history_file = os.path.join(config["logs_dir"], "evaluation", "report_history.csv")
    if os.path.exists(history_file):
        df = pd.read_csv(history_file, on_bad_lines='skip')
        st.dataframe(df, width='stretch')
    else:
        st.warning("⚠️ Nenhum histórico encontrado.")

# ======================================================
# Histórico geral exportável
# ======================================================
if st.session_state.history and show_history:
    st.markdown("### 📥 Exportar Histórico Geral")
    if st.button("Gerar CSV", key="export_csv"):
        all_records = []
        for r in st.session_state.history:
            for d in r.get("details", []):
                all_records.append({
                    "modo": r.get("modo"),
                    "arquivo": r.get("file"),
                    "texto": r.get("text", r.get("predicted")),
                    "true_label": r.get("true_label", ""),
                    "posição": d.get("position"),
                    "caractere": d.get("char"),
                    "tipo": d.get("type"),
                    "confiança": d.get("confidence"),
                    "acuracia": r.get("acuracia", np.nan) 
                })
        df_export = pd.DataFrame(all_records)
        st.download_button(
            "⬇️ Baixar CSV",
            df_export.to_csv(index=False).encode("utf-8"),
            "historico_predicoes.csv"
        )
