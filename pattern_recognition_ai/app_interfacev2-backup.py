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
tab1, tab2, tab3, tab4 = st.tabs(["📤 Envio Manual", "🧪 Teste com Dataset Local", "🧪 Debug", "📊 Relatório de Progresso"])

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

        # Mostrar detalhes
        st.markdown("### 🔠 Detalhes da Predição por Caractere")
        if details:
            df_details = pd.DataFrame(details)

            # Adicionar colunas de porcentagem
            if "confidence" in df_details.columns:
                df_details["confidence (%)"] = df_details["confidence"] * 100
            if "confidence_alpha" in df_details.columns:
                df_details["confidence_alpha (%)"] = df_details["confidence_alpha"] * 100
            if "confidence_number" in df_details.columns:
                df_details["confidence_number (%)"] = df_details["confidence_number"] * 100

            # Exibir
            cols_to_show = ["position", "char", "type", "confidence (%)"]
            if "confidence_alpha (%)" in df_details.columns:
                cols_to_show.append("confidence_alpha (%)")
            if "confidence_number (%)" in df_details.columns:
                cols_to_show.append("confidence_number (%)")
            
            st.dataframe(df_details[cols_to_show], use_container_width=True)

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

        # Mostrar detalhes por caractere
        st.markdown("### 🔠 Detalhes por Caractere")
        if details:
            df_details = pd.DataFrame(details)

            if "confidence" in df_details.columns:
                df_details["confidence (%)"] = df_details["confidence"] * 100
            if "confidence_alpha" in df_details.columns:
                df_details["confidence_alpha (%)"] = df_details["confidence_alpha"] * 100
            if "confidence_number" in df_details.columns:
                df_details["confidence_number (%)"] = df_details["confidence_number"] * 100

            cols_to_show = ["position", "char", "type", "confidence (%)"]
            if "confidence_alpha (%)" in df_details.columns:
                cols_to_show.append("confidence_alpha (%)")
            if "confidence_number (%)" in df_details.columns:
                cols_to_show.append("confidence_number (%)")
            
            st.dataframe(df_details[cols_to_show], use_container_width=True)

with tab3:
    st.header("🤖 Treinamento Assistido - Debug / Ensino da IA")
    
    uploaded_file = st.file_uploader("Envie uma imagem CAPTCHA", type=["png", "jpg", "jpeg"], key="train_upload")
    
    if uploaded_file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        predictor_debug = load_model(debug=True)
        result, debug_steps, details = predictor_debug.predict_with_debug(img_path)


        st.success(f"🔍 Resultado do modelo: **{result}**")

        # Mostrar detalhes por caractere
        if hasattr(predictor_debug, "last_details") and predictor_debug.last_details:
            df_debug_details = pd.DataFrame(predictor_debug.last_details)

            if "confidence" in df_debug_details.columns:
                df_debug_details["confidence (%)"] = df_debug_details["confidence"] * 100
            if "confidence_alpha" in df_debug_details.columns:
                df_debug_details["confidence_alpha (%)"] = df_debug_details["confidence_alpha"] * 100
            if "confidence_number" in df_debug_details.columns:
                df_debug_details["confidence_number (%)"] = df_debug_details["confidence_number"] * 100

            cols_to_show = ["position", "char", "type", "confidence (%)"]
            if "confidence_alpha (%)" in df_debug_details.columns:
                cols_to_show.append("confidence_alpha (%)")
            if "confidence_number (%)" in df_debug_details.columns:
                cols_to_show.append("confidence_number (%)")

            st.markdown("### 🔠 Predição por Caractere")
            st.dataframe(df_debug_details[cols_to_show], use_container_width=True)

        # Mostrar histórico de imagens intermediárias
        with st.expander("🧩 Histórico de processamento da imagem"):
            cols = st.columns(3)
            for i, (step, img) in enumerate(debug_steps.items()):
                if img is None:
                    continue
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = Image.fromarray(img)
                elif not isinstance(img, Image.Image):
                    continue
                cols[i % 3].image(img, caption=step, use_column_width=True)

        # 🔧 Correção manual / treino assistido
        st.markdown("### ✏️ Corrigir / Confirmar Predição")
        corrected_text = st.text_input("Texto correto do CAPTCHA:", value=result, key="corrected_text")

        if st.button("💾 Salvar Correção"):
            st.session_state.history.append({
                "modo": "train_assisted",
                "file": uploaded_file.name,
                "predicted": result,
                "corrected": corrected_text,
                "details": predictor_debug.last_details
            })
            st.success("✅ Correção salva com sucesso!")


# ======================================================
# 📊 ABA 4 – Relatório de Progresso
# ======================================================
with tab4:
    st.header("📊 Relatório de Progresso do Modelo de IA")

    history_file = os.path.join(config["logs_dir"], "evaluation", "report_history.csv")
    if not os.path.exists(history_file):
        st.warning("⚠️ Nenhum histórico encontrado. Execute avaliações primeiro.")
    else:
        try:
            df = pd.read_csv(history_file, on_bad_lines='skip')
        except Exception as e:
            st.error(f"Erro ao ler o histórico: {e}")
            df = pd.DataFrame()

        if not df.empty:
            # Converter colunas numéricas
            df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
            df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
            df["image_count"] = pd.to_numeric(df["image_count"], errors="coerce")
            df = df.dropna(subset=["accuracy", "loss", "image_count"])

            df["progress_percentage"] = df["accuracy"] / 0.98 * 100
            df["Tipo"] = df.get("type", pd.Series("unknown"))
            df["Subtipo"] = df.get("subtype", pd.Series("unknown"))

            grouped = df.groupby(["Tipo", "Subtipo"]).agg(
                Imagens=("image_count", "sum"),
                Acurácia_Média=("accuracy", "mean"),
                Loss_Médio=("loss", "mean"),
                Progresso_Médio=("progress_percentage", "mean")
            ).reset_index()

            st.markdown("### 📌 Métricas por Tipo e Subtipo")
            st.dataframe(grouped.style.format({
                "Acurácia_Média": "{:.2%}",
                "Loss_Médio": "{:.4f}",
                "Progresso_Médio": "{:.2f}"
            }), use_container_width=True)

            # Média total ponderada
            total_images = grouped["Imagens"].sum()
            weighted_accuracy = (grouped["Acurácia_Média"] * grouped["Imagens"]).sum() / total_images
            weighted_loss = (grouped["Loss_Médio"] * grouped["Imagens"]).sum() / total_images
            weighted_progress = (grouped["Progresso_Médio"] * grouped["Imagens"]).sum() / total_images

            total_df = pd.DataFrame([{
                "Total Imagens": total_images,
                "Acurácia Média": weighted_accuracy,
                "Loss Média": weighted_loss,
                "Progresso Médio (%)": weighted_progress
            }])

            st.markdown("### 📊 Média Total")
            st.dataframe(total_df.style.format({
                "Acurácia Média": "{:.2%}",
                "Loss Média": "{:.4f}",
                "Progresso Médio (%)": "{:.2f}"
            }), use_container_width=True)

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
                    "tipo_detectado": d.get("type"),
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
