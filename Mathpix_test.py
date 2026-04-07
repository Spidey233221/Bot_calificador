import requests
import base64
import json
import os
import re

# ════════════════════════════════════
#  Pon aquí tus credenciales de Mathpix
# ════════════════════════════════════
MATHPIX_APP_ID = "edtechsa_edabfb_05c62a"
MATHPIX_APP_KEY = "b3472076544729f5a892d8ecb150e7b9205ab308c337b5d3aef69ba804cf61ab"

FOLDER = "Prueba"

# ════════════════════════════════════
# OCR
# ════════════════════════════════════
def ocr_image(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
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
            "formats": ["text", "latex_styled"],
            "data_options": {
                "include_asciimath": True,
                "include_latex": True
            }
        }
    )

    return response.json()


# ════════════════════════════════════
# PARSEO DE EJERCICIOS
# ════════════════════════════════════
def parse_exercises(mathpix_text):
    pattern = r"\((\d+)\)(.*?)(?=\(\d+\)|$)"
    matches = re.findall(pattern, mathpix_text, re.DOTALL)

    ejercicios = {}

    for num, contenido in matches:
        num = int(num)

        # Intentar obtener resultado después de "="
        resultado_match = re.search(r"=\s*([-\d\.]+)", contenido)

        if resultado_match:
            resultado = resultado_match.group(1)
        else:
            # Si no hay "=", intentar evaluar la expresión
            try:
                expr = contenido.split("=")[0].strip()
                expr = expr.replace("×", "*").replace("÷", "/")
                resultado = str(eval(expr))
            except:
                resultado = None

        if resultado is not None:
            ejercicios[num] = resultado.strip()

    return ejercicios


# ════════════════════════════════════
# JSON RESPUESTAS
# ════════════════════════════════════
def parse_json_respuestas(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    respuestas = {}

    # 👇 ahora entramos a answers_raw
    answers = data.get("answers_raw", [])

    for item in answers:
        index = item["index"] + 1  # 👈 clave: alinear con (1), (2), ...
        respuestas[index] = item["text"]

    return respuestas


# ════════════════════════════════════
# COMPARACIÓN
# ════════════════════════════════════
def comparar(ejercicios, respuestas):
    resultados = {}

    for i in ejercicios:
        correcto = ejercicios[i]
        usuario = respuestas.get(i)

        resultados[i] = {
            "correcto": correcto,
            "usuario": usuario,
            "es_correcto": str(correcto) == str(usuario)
        }

    return resultados


# ════════════════════════════════════
# MAIN
# ════════════════════════════════════
def run():
    files = ["Prueba1"]

    # Buscar archivos
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

    # Cargar respuestas JSON
    json_respuestas_path = os.path.join(FOLDER, "respuestas.json")
    if os.path.exists(json_respuestas_path):
        respuestas = parse_json_respuestas(json_respuestas_path)
        print("✅ respuestas.json cargado\n")
    else:
        respuestas = None
        print("⚠ No se encontró respuestas.json\n")

    all_results = {}

    for filepath in found_files:
        filename = os.path.basename(filepath)

        print(f"{'='*40}")
        print(f"🔍 Procesando: {filename}")
        print(f"{'='*40}")

        result = ocr_image(filepath)

        text = result.get("text", "")
        latex = result.get("latex_styled", "")
        confidence = result.get("confidence", 0)
        error = result.get("error", "")

        if error:
            print(f"❌ Error: {error}\n")
            continue

        print(f"\n📊 Confianza: {confidence:.2%}")

        print(f"\n📝 Texto detectado:")
        print(f"{text}")

        print(f"\n🔢 LaTeX detectado:")
        print(f"{latex}")

        # ════════════════════════════════════
        # VALIDACIÓN
        # ════════════════════════════════════
        if respuestas:
            ejercicios = parse_exercises(text)
            comparacion = comparar(ejercicios, respuestas)

            print("\n🧪 Validación de resultados:\n")

            for i, data in comparacion.items():
                estado = "✅" if data["es_correcto"] else "❌"
                print(
                    f"Ejercicio ({i}): "
                    f"Correcto={data['correcto']} | "
                    f"Usuario={data['usuario']} -> {estado}"
                )

        print()

        all_results[filename] = {
            "text": text,
            "latex": latex,
            "confidence": confidence,
            "raw": result
        }

    # Guardar OCR
    output_path = os.path.join(FOLDER, "ocr_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"{'='*40}")
    print("✅ Proceso terminado")
    print(f"📁 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    run()
