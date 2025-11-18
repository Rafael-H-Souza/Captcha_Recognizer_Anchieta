import cv2
from pathlib import Path
from src.preprocessing.filters import preprocess_image  # ajuste o path conforme necessário

def generate_preprocessed_images():
    input_dir = Path("data/raw")
    output_dir = Path("debug/processed_imgs")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.png")) + \
                  list(input_dir.glob("*.jpg")) + \
                  list(input_dir.glob("*.jpeg"))

    if not image_files:
        print("⚠️ Nenhuma imagem encontrada em data/raw/")
        return

    for img_path in image_files:
        try:
            processed = preprocess_image(str(img_path))
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), processed)
            print(f"✅ {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"❌ Erro com {img_path.name}: {e}")

    print(f"\n🎯 Imagens pré-processadas salvas em: {output_dir.resolve()}")

if __name__ == "__main__":
    generate_preprocessed_images()
