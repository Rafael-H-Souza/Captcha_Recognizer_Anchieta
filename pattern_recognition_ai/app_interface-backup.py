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
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🤖 Validador Automático de CAPTCHA")

# ======================================================
# 🧠 Inicialização do Modelo (cacheado para desempenho)
# ======================================================
@st.cache_resource
def load_model(debug: bool = False):
    model_path = config["final_model_path"]
    return Predictor(model_path, debug=debug)

predictor = load_model()

# ======================================================
# 🔖 Organização por Abas
# ======================================================
tab1, tab2, tab3 = st.tabs(["📤 Envio Manual", "🧪 Teste com Dataset Local", "🧪 Debug"])

# ======================================================
# 🧾 Histórico Global de Predições
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# 📤 ABA 1 – Envio Manual de CAPTCHA
# ======================================================
with tab1:
    st.markdown("""
    Envie uma imagem de CAPTCHA para que o modelo de IA tente **decodificá-la**.  
    O sistema exibirá o **texto previsto** e os **detalhes de cada caractere**, com nível de confiança.
    """)

    uploaded_file = st.file_uploader("Selecione a imagem de CAPTCHA:", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Imagem enviada", use_column_width=True)

        temp_dir = "data/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(temp_path)

        with st.spinner("🔄 Processando imagem com IA..."):
            decoded_text, details = predictor.predict_with_details(temp_path)

        st.success(f"✅ Texto decodificado: **{decoded_text}**")
        st.markdown("### 🔠 Detalhes da Predição por Caractere")
        st.dataframe(pd.DataFrame(details), use_container_width=True)

        st.session_state.history.append({
            "modo": "upload",
            "file": uploaded_file.name,
            "text": decoded_text,
            "details": details
        })

# ======================================================
# 🧪 ABA 2 – Teste com Dataset Local
# ======================================================
with tab2:
    BASE_PATH = "imagens/archive/samples"

    if not os.path.exists(BASE_PATH):
        st.warning("⚠️ Pasta de imagens não encontrada. Verifique o caminho BASE_PATH.")
    else:
        st.markdown("""
        Teste o modelo com imagens locais de CAPTCHA já conhecidas.  
        O sistema compara o **texto real (do nome do arquivo)** com a **predição do modelo**.
        """)

        image_files = sorted([f for f in os.listdir(BASE_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        selected_image = st.selectbox("📂 Escolha uma imagem", image_files)
        image_path = os.path.join(BASE_PATH, selected_image)

        image = Image.open(image_path)
        st.image(image, caption=f"Imagem: {selected_image}", width=300)

        true_label = os.path.splitext(selected_image)[0]

        with st.spinner("🧠 Gerando predição..."):
            predicted_text, details = predictor.predict_with_details(image_path)
            confidence = sum(d.get("confidence", 0) for d in details) / len(details) if details else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Texto Real", true_label)
        col2.metric("Predição", predicted_text)
        col3.metric("Confiança (%)", f"{confidence*100:.2f}")

        acertou = predicted_text.lower() == true_label.lower()
        st.write(f"**Resultado:** {'✅ Acertou!' if acertou else '❌ Errou!'}")

        if "results" not in st.session_state:
            st.session_state.results = []

        if st.button("💾 Salvar Resultado"):
            st.session_state.results.append({
                "Imagem": selected_image,
                "Verdadeiro": true_label,
                "Previsto": predicted_text,
                "Confiança": f"{confidence*100:.2f}%",
                "Acertou": "✅" if acertou else "❌"
            })

        if st.session_state.results:
            st.subheader("📊 Histórico de Resultados (Dataset Local)")
            st.dataframe(pd.DataFrame(st.session_state.results), use_container_width=True)

# ======================================================
# 💾 ABA 3 – Debug
# ======================================================
with tab3:
    st.header("📸 Envio Manual (Debug)")
    uploaded_file = st.file_uploader("Envie uma imagem CAPTCHA", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        predictor_debug = load_model(debug=True)
        result, debug_steps = predictor_debug.predict_with_debug(img_path)

        st.success(f"🔍 Resultado: **{result}**")

        with st.expander("🧩 Histórico de processamento da imagem (modo debug)"):
            cols = st.columns(3)
            for i, (step, img) in enumerate(debug_steps.items()):
                # validação do tipo da imagem
                if img is None:
                    st.warning(f"⚠️ Etapa '{step}' não retornou imagem válida.")
                    continue
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 2:  # grayscale → RGB
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = Image.fromarray(img)
                elif not isinstance(img, Image.Image):
                    st.warning(f"⚠️ Tipo inválido em '{step}': {type(img)}")
                    continue

                cols[i % 3].image(img, caption=step, use_column_width=True)


# ======================================================
# 💾 Exportar Histórico Geral
# ======================================================
if st.session_state.history:
    st.markdown("### 📥 Exportar Histórico Geral")
    if st.button("Gerar CSV"):
        all_records = []
        for r in st.session_state.history:
            for d in r["details"]:
                all_records.append({
                    "modo": r["modo"],
                    "arquivo": r["file"],
                    "texto_completo": r["text"],
                    "posição": d.get("position"),
                    "caractere": d.get("char"),
                    "confiança": d.get("confidence")
                })
        df_export = pd.DataFrame(all_records)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Clique para baixar histórico CSV",
            data=csv,
            file_name="historico_predicoes.csv",
            mime="text/csv"
        )
