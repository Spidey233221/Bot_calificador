import requests
import base64
import json
import os

# ════════════════════════════════════
#  Pon aquí tus credenciales de Mathpix
# ════════════════════════════════════
MATHPIX_APP_ID = ""
MATHPIX_APP_KEY = ""

FOLDER = "Prueba"


def ocr_image(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    # Detectar extensión
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".jpg" or ext == ".jpeg":
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        mime = "image/png"

    response = requests.post(
        "https://api.mathpix.com/v3/text",
        headers={
            "app_id": MATHPIX_APP_ID,
            "app_key": MATHPIX_APP_KEY,
            "Content-type": "application/json"
        },
        json={
            "src": f"data:{mime};base64,{image_data}",
            "formats": [
                "text",
                "latex_styled"
            ],
            "data_options": {
                "include_asciimath": True,
                "include_latex": True
            }
        }
    )

    return response.json()


def run():
    files = ["Prueba1", "Prueba2", "Prueba3"]

    # Buscar la extensión correcta
    found_files = []
    for name in files:
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(FOLDER, name + ext)
            if os.path.exists(path):
                found_files.append(path)
                break
        else:
            print(f"⚠ No encontré {name} en {FOLDER}")

    if not found_files:
        print("❌ No se encontraron imágenes")
        return

    print(f"📁 Carpeta: {FOLDER}")
    print(f"📄 Imágenes encontradas: {len(found_files)}\n")

    all_results = {}

    for filepath in found_files:
        filename = os.path.basename(filepath)
        print(f"{'='*40}")
        print(f"🔍 Procesando: {filename}")
        print(f"{'='*40}")

        result = ocr_image(filepath)

        # Mostrar resultados
        text = result.get("text", "")
        latex = result.get("latex_styled", "")
        confidence = result.get("confidence", 0)
        error = result.get("error", "")

        if error:
            print(f"❌ Error: {error}\n")
            continue

        print(f"\n📊 Confianza: {confidence:.2%}")

        print(f"\n📝 Texto detectado:")
        print(f"   {text}")

        print(f"\n🔢 LaTeX detectado:")
        print(f"   {latex}")

        print()

        all_results[filename] = {
            "text": text,
            "latex": latex,
            "confidence": confidence,
            "raw": result
        }

    # Guardar todo en JSON
    output_path = os.path.join(
        FOLDER, "ocr_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            all_results, f,
            indent=2, ensure_ascii=False
        )

    print(f"{'='*40}")
    print(f"✅ Proceso terminado")
    print(f"📁 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    run()