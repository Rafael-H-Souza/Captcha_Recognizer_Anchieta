import os
import streamlit as st
from PIL import Image
import pandas as pd
from src.config.settings import config
from src.inference.predictor import Predictor
import numpy as np
import cv2

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
debug_mode = st.sidebar.checkbox("Modo Debug", value=False)
show_history = st.sidebar.checkbox("Mostrar Histórico de Predições", value=True)

# ======================================================
# 🧠 Carregar modelo (cacheado)
# ======================================================
@st.cache_resource
def load_model(debug=False):
    return Predictor(config["final_model_path"], debug=debug)

predictor = load_model(debug=debug_mode)

# ======================================================
# Sessão de Histórico
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
    uploaded_file = st.file_uploader("Selecione uma imagem:", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem enviada", use_column_width=True)

        temp_dir = "data/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(temp_path)

        with st.spinner("🔄 Processando imagem..."):
            decoded_text, details = predictor.predict_with_details(temp_path)

        st.success(f"✅ Texto decodificado: **{decoded_text}**")

        # Mostrar detalhes por caractere em tabela
        if details:
            df_details = pd.DataFrame(details)
            for col in ["confidence", "confidence_alpha", "confidence_number"]:
                if col in df_details.columns:
                    df_details[col] = df_details[col] * 100

            st.markdown("#### 🔠 Detalhes por Caractere")
            st.dataframe(df_details, use_container_width=True)

        st.session_state.history.append({
            "modo": "upload",
            "file": uploaded_file.name,
            "text": decoded_text,
            "details": details
        })

# ======================================================
# ABA 2 - Teste com Dataset Local
# ======================================================
with tab2:
    st.subheader("🧪 Teste com Dataset Local")
    BASE_PATH = "imagens/archive/samples"
    if not os.path.exists(BASE_PATH):
        st.warning("⚠️ Pasta de imagens não encontrada.")
    else:
        image_files = sorted([f for f in os.listdir(BASE_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        selected_image = st.selectbox("Escolha uma imagem", image_files)
        image_path = os.path.join(BASE_PATH, selected_image)
        image = Image.open(image_path)
        st.image(image, caption=f"Imagem: {selected_image}", width=300)

        true_label = os.path.splitext(selected_image)[0]

        predicted_text, details = predictor.predict_with_details(image_path)
        confidence = np.mean([d.get("confidence", 0) for d in details]) if details else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Texto Real", true_label)
        col2.metric("Predição", predicted_text)
        col3.metric("Confiança (%)", f"{confidence*100:.2f}")

        st.write(f"**Resultado:** {'✅ Acertou!' if predicted_text.lower()==true_label.lower() else '❌ Errou!'}")

        # Detalhes
        if details:
            df_details = pd.DataFrame(details)
            for col in ["confidence", "confidence_alpha", "confidence_number"]:
                if col in df_details.columns:
                    df_details[col] = df_details[col]*100
            st.markdown("#### 🔠 Detalhes por Caractere")
            st.dataframe(df_details, use_container_width=True)

# ======================================================
# ABA 3 - Debug / Ensino
# ======================================================
with tab3:
    st.subheader("🧠 Treinamento Assistido / Debug")
    uploaded_file = st.file_uploader("Envie CAPTCHA (debug)", type=["png","jpg","jpeg"], key="debug_upload")
    if uploaded_file:
        temp_dir = "data/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path,"wb") as f: f.write(uploaded_file.read())

        predictor_debug = load_model(debug=True)
        result, debug_steps, details = predictor_debug.predict_with_debug(img_path)
        st.success(f"🔍 Resultado do modelo: **{result}**")

        # Grid de imagens intermediárias
        with st.expander("🖼 Histórico de Processamento"):
            cols = st.columns(3)
            for i,(step,img) in enumerate(debug_steps.items()):
                if isinstance(img,np.ndarray):
                    if len(img.shape)==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                    img=Image.fromarray(img)
                elif not isinstance(img,Image.Image): continue
                cols[i%3].image(img, caption=step, use_column_width=True)

        # Correção manual
        corrected_text = st.text_input("Texto correto:", value=result, key="corrected_text")
        if st.button("💾 Salvar Correção"):
            st.session_state.history.append({
                "modo":"debug",
                "file":uploaded_file.name,
                "predicted":result,
                "corrected":corrected_text,
                "details":details
            })
            st.success("✅ Correção salva!")

# ======================================================
# ABA 4 - Relatório de Progresso
# ======================================================
with tab4:
    st.subheader("📊 Relatório de Progresso")
    history_file = os.path.join(config["logs_dir"], "evaluation", "report_history.csv")
    if os.path.exists(history_file):
        df = pd.read_csv(history_file, on_bad_lines='skip')
        df["accuracy"]=pd.to_numeric(df["accuracy"],errors="coerce")
        df["loss"]=pd.to_numeric(df["loss"],errors="coerce")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ Nenhum histórico encontrado.")

# ======================================================
# Histórico geral exportável
# ======================================================
if st.session_state.history and show_history:
    st.markdown("### 📥 Exportar Histórico Geral")
    if st.button("Gerar CSV"):
        all_records=[]
        for r in st.session_state.history:
            for d in r["details"]:
                all_records.append({
                    "modo": r["modo"],
                    "arquivo": r["file"],
                    "texto": r.get("text",r.get("predicted")),
                    "posição": d.get("position"),
                    "caractere": d.get("char"),
                    "tipo": d.get("type"),
                    "confiança": d.get("confidence")
                })
        df_export=pd.DataFrame(all_records)
        st.download_button("⬇️ Baixar CSV", df_export.to_csv(index=False).encode("utf-8"), "historico_predicoes.csv")
