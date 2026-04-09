from playwright.sync_api import sync_playwright
import os
import time
import re
import json
import requests
import base64
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════
BASE_DIR = "base_data"

# Credenciales Mathpix
MATHPIX_APP_ID = "edtechsa_edabfb_05c62a"
MATHPIX_APP_KEY = "b3472076544729f5a892d8ecb150e7b9205ab308c337b5d3aef69ba804cf61ab"

# Configuración de recorte
CROP_BOTTOM_PIXELS = 40   # Footer Kumon (©2022...)

# Configuración de preprocesamiento de imagen
PREPROCESS_STRIP_RED = True    # Borrar X rojas y cuadros naranjas
PREPROCESS_UPSCALE = 2         # Escalar imagen 2x para mejor OCR
PREPROCESS_CONTRAST = 1.8      # Factor de contraste

# ════════════════════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO DE IMAGEN (Pillow)
# ════════════════════════════════════════════════════════════════════════════

def remove_red_ink(pil_rgb_image, red_threshold=130, dominance=25, dilate=5):
    """
    Elimina todo pixel rojo/naranja de la imagen reemplazandolo por blanco.
    Detecta: X rojas, cuadros naranjas, indicador 40 central, etc.
    """
    arr = np.array(pil_rgb_image)
    R = arr[:, :, 0].astype(int)
    G = arr[:, :, 1].astype(int)
    B = arr[:, :, 2].astype(int)

    red_dominance = R - np.maximum(G, B)
    reddish = (R > red_threshold) & (red_dominance > dominance)

    if dilate and dilate > 1:
        mask_img = Image.fromarray((reddish * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(dilate))
        reddish = np.array(mask_img) > 128

    result = arr.copy()
    result[reddish] = [255, 255, 255]
    return Image.fromarray(result)


def preprocess_image(input_path, output_path=None):
    """
    Preprocesa imagen antes de enviar a Mathpix:
    1. Borra rojo/naranja (X, cuadros, indicador central)
    2. Grises + autocontrast + contraste
    3. Upscale 2x
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_pp{ext}"

    img = Image.open(input_path).convert("RGB")

    if PREPROCESS_STRIP_RED:
        img = remove_red_ink(img)

    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Contrast(img).enhance(PREPROCESS_CONTRAST)

    if PREPROCESS_UPSCALE and PREPROCESS_UPSCALE != 1:
        new_size = (img.width * PREPROCESS_UPSCALE, img.height * PREPROCESS_UPSCALE)
        img = img.resize(new_size, Image.LANCZOS)

    img.save(output_path, "PNG", optimize=True)
    return output_path


# ════════════════════════════════════════════════════════════════════════════
#  FUNCIONES OCR (Mathpix)
# ════════════════════════════════════════════════════════════════════════════

def ocr_image(image_path):
    """Envia imagen a Mathpix (con preprocesamiento)."""
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        return {"error": "Credenciales Mathpix no configuradas"}

    try:
        path_to_send = preprocess_image(image_path)
    except Exception as e:
        print(f"         ! Preprocesamiento fallo: {e}, usando original")
        path_to_send = image_path

    with open(path_to_send, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post(
            "https://api.mathpix.com/v3/text",
            headers={
                "app_id": MATHPIX_APP_ID,
                "app_key": MATHPIX_APP_KEY,
                "Content-type": "application/json"
            },
            json={
                "src": f"data:image/png;base64,{image_data}",
                "formats": ["text"],
                "rm_spaces": True,
                "rm_fonts": True,
                "math_inline_delimiters": ["", ""],
                "math_display_delimiters": ["", ""],
                "include_line_data": True,
            },
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def parse_kumon_exercises(ocr_text, expected_numbers=None):
    """
    Parser de ejercicios Kumon. Busca patrones: (N) A +/- B = R.
    Si se pasa expected_numbers (del DOM), solo acepta esos numeros.
    """
    ejercicios = {}

    text = ocr_text
    text = re.sub(r'\\[\(\)\[\]]', '', text)
    text = re.sub(r'\\quad', ' ', text)
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'[\u25a1\u25a0\u2612\u2611\u2713\u2717\u00d7]', '', text)
    text = re.sub(r'\bx\b(?!\d)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'kum[@o]n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sumas?\s*\([^)]*\)[^(]*', '', text)
    text = re.sub(r'Resta\.?', '', text)
    text = re.sub(r'Suma\.?', '', text)
    text = re.sub(r'A\d+[ab]?', '', text)
    text = re.sub(r'\s+', ' ', text)

    expected_set = set(expected_numbers) if expected_numbers else None

    # Metodo 1: Patron completo (N) op1 +/- op2 = resultado
    pattern1 = r'\((\d+)\)\s*(\d+)\s*([\+\-])\s*(\d+)\s*=\s*(\d+)'
    for match in re.finditer(pattern1, text):
        num = int(match.group(1))
        if expected_set and num not in expected_set:
            continue
        ejercicios[num] = {
            "expresion": f"{match.group(2)} {match.group(3)} {match.group(4)}",
            "resultado_alumno": match.group(5),
            "confianza": "alta"
        }

    # Metodo 2: Por lineas para los faltantes
    if expected_set:
        faltantes = expected_set - set(ejercicios.keys())
    else:
        faltantes = None if len(ejercicios) >= 5 else set()

    if faltantes is None or faltantes:
        lines = text.split('\n')
        for line in lines:
            num_match = re.search(r'\((\d+)\)', line)
            if not num_match:
                continue
            num = int(num_match.group(1))
            if num in ejercicios:
                continue
            if expected_set and num not in expected_set:
                continue

            eq_match = re.search(r'=\s*(\d+)', line)
            if eq_match:
                resultado = eq_match.group(1)
                after_num = line[num_match.end():]
                expr_match = re.search(r'(\d+)\s*([\+\-])\s*(\d+)', after_num)
                expresion = (f"{expr_match.group(1)} {expr_match.group(2)} {expr_match.group(3)}"
                             if expr_match else "?")
                ejercicios[num] = {
                    "expresion": expresion,
                    "resultado_alumno": resultado,
                    "confianza": "media"
                }

    # Metodo 3: Ultra flexible (solo si no hay expected)
    if expected_set is None and len(ejercicios) < 3:
        pattern3 = r'\((\d+)\)[^=]*=\s*(\d+)'
        for num_str, resultado in re.findall(pattern3, text):
            num = int(num_str)
            if num not in ejercicios:
                ejercicios[num] = {
                    "expresion": "?",
                    "resultado_alumno": resultado,
                    "confianza": "baja"
                }

    # Marcar faltantes como no detectados
    if expected_set:
        for num in expected_set:
            if num not in ejercicios:
                ejercicios[num] = {
                    "expresion": "?",
                    "resultado_alumno": None,
                    "confianza": "no_detectado"
                }

    return ejercicios


# Confusiones manuscritas comunes del OCR
CONFUSIONES_OCR = {
    '1': ['7', '4', 'l', 'i', '|'],
    '7': ['1', '2'],
    '0': ['6', '8', '9', 'o', 'O'],
    '3': ['8', '5'],
    '5': ['6', '8', '3', 's', 'S'],
    '6': ['0', '8', 'b', '5'],
    '8': ['0', '3', '6', 'B', '9'],
    '9': ['4', 'g', 'q', '8'],
    '4': ['9', '1', '7'],
    '2': ['7', 'z', 'Z'],
}


def es_error_ocr_probable(alumno_str, correcto_str):
    """Detecta si la diferencia entre alumno y correcto es confusion OCR conocida."""
    if not alumno_str or not correcto_str or alumno_str == correcto_str:
        return False, None

    if len(alumno_str) != len(correcto_str):
        if abs(len(alumno_str) - len(correcto_str)) == 1:
            if correcto_str in alumno_str or alumno_str in correcto_str:
                return True, "digito extra/faltante"
        return False, None

    diferencias = 0
    notas = []
    for a, c in zip(alumno_str, correcto_str):
        if a != c:
            diferencias += 1
            if c in CONFUSIONES_OCR and a in CONFUSIONES_OCR[c]:
                notas.append(f"{c}->{a}")
            elif a in CONFUSIONES_OCR and c in CONFUSIONES_OCR[a]:
                notas.append(f"{c}->{a}")

    if diferencias == 1 and notas:
        return True, notas[0]
    if diferencias <= 2 and len(notas) == diferencias:
        return True, ", ".join(notas)
    return False, None


def comparar_respuestas(ejercicios_ocr, respuestas_correctas):
    """Compara respuestas con validacion fuzzy."""
    resultados = []

    for num in sorted(set(ejercicios_ocr.keys()) | set(respuestas_correctas.keys())):
        ej = ejercicios_ocr.get(num, {})
        expresion = ej.get("expresion", "?")
        alumno = ej.get("resultado_alumno")
        correcto = respuestas_correctas.get(num)

        alumno_str = str(alumno).strip() if alumno else ""
        correcto_str = str(correcto).strip() if correcto else ""

        if not alumno_str or not correcto_str:
            es_correcto, es_fuzzy, nota_fuzzy = None, False, None
        elif alumno_str == correcto_str:
            es_correcto, es_fuzzy, nota_fuzzy = True, False, None
        else:
            es_fuzzy, nota_fuzzy = es_error_ocr_probable(alumno_str, correcto_str)
            es_correcto = False

        resultados.append({
            "numero": num,
            "expresion": expresion,
            "alumno": alumno if alumno else "-",
            "correcto": correcto if correcto else "-",
            "es_correcto": es_correcto,
            "es_fuzzy": es_fuzzy,
            "nota_fuzzy": nota_fuzzy
        })

    return resultados


def mostrar_validacion(resultados, titulo=""):
    """Muestra resultados con formato visual."""
    if titulo:
        print(f"\n  [{titulo}]")
    print("  " + "-" * 80)

    correctos, incorrectos, fuzzy, sin_respuesta = 0, 0, 0, 0

    for r in resultados:
        if r["es_correcto"] is None:
            estado = "?"
            sin_respuesta += 1
        elif r["es_correcto"]:
            estado = "OK"
            correctos += 1
        elif r.get("es_fuzzy"):
            estado = f"revisar ({r['nota_fuzzy']})" if r.get("nota_fuzzy") else "revisar"
            fuzzy += 1
        else:
            estado = "X"
            incorrectos += 1

        expr = r["expresion"][:12].ljust(12)
        print(
            f"  Ejercicio ({r['numero']:2d}): {expr} | "
            f"Alumno: {str(r['alumno']):6s} | "
            f"Correcto: {str(r['correcto']):6s} {estado}"
        )

    print("  " + "-" * 80)
    total = correctos + incorrectos + fuzzy
    if total > 0:
        efectivos = correctos + fuzzy
        print(f"  Resultado: {correctos} correctos", end="")
        if fuzzy > 0: print(f" + {fuzzy} probables (revisar)", end="")
        if incorrectos > 0: print(f" + {incorrectos} incorrectos", end="")
        print(f" = {efectivos}/{total} ({100*efectivos/total:.1f}%)")
        if sin_respuesta > 0:
            print(f"      {sin_respuesta} ejercicios sin respuesta detectada")
    else:
        print(f"  No se pudieron comparar respuestas")

    return correctos, total, fuzzy


# ════════════════════════════════════════════════════════════════════════════
#  FUNCIONES PLAYWRIGHT
# ════════════════════════════════════════════════════════════════════════════

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


def parse_level_page(user_input):
    user_input = user_input.strip().upper()
    match = re.match(r'^([A-Z0-9]+?)(\d+)$', user_input)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def get_visible_worksheet_pages(page):
    visible_pages = []
    pages = page.locator(".worksheet-group-page")
    count = pages.count()
    viewport = page.viewport_size or {"width": 1600, "height": 1000}

    for i in range(count):
        try:
            item = pages.nth(i)
            bb = item.bounding_box()
            if not bb or bb["width"] < 150 or bb["height"] < 200:
                continue
            inter_w = max(0, min(bb["x"] + bb["width"], viewport["width"]) - max(bb["x"], 0))
            inter_h = max(0, min(bb["y"] + bb["height"], viewport["height"]) - max(bb["y"], 0))
            ratio = (inter_w * inter_h) / (bb["width"] * bb["height"])
            if ratio > 0.4:
                visible_pages.append({"index": i, "bbox": bb, "element": item})
        except:
            pass

    visible_pages.sort(key=lambda v: (v["bbox"]["y"], v["bbox"]["x"]))
    return visible_pages


def smart_click(locator, name):
    for method in ['normal', 'force', 'mouse']:
        try:
            if method == 'normal':
                locator.click(timeout=3000)
            elif method == 'force':
                locator.click(force=True, timeout=3000)
            else:
                bb = locator.bounding_box()
                if bb:
                    locator.page.mouse.click(bb["x"] + bb["width"]/2, bb["y"] + bb["height"]/2)
            return True
        except:
            pass
    return False


def activate_double_view(page):
    btn = page.locator("#BothSidesDisplayButton")
    smart_click(btn, "BothSidesDisplayButton")
    page.wait_for_timeout(1500)


def go_next_page(page):
    btn = page.locator("button.down.pager-button")
    try:
        if btn.is_disabled() or "disabled" in (btn.get_attribute("class") or ""):
            return False
    except:
        pass
    ok = smart_click(btn, "Down pager button")
    page.wait_for_timeout(2000)
    return ok


def get_page_fingerprint(page):
    visible = get_visible_worksheet_pages(page)
    if not visible:
        return None, visible
    return tuple((v["index"], round(v["bbox"]["y"], 1)) for v in visible), visible


def check_page_has_checkboxes(page, bbox):
    mark_boxes = page.locator(".mark-box")
    found = 0
    try:
        for i in range(mark_boxes.count()):
            box = mark_boxes.nth(i)
            if not box.is_visible():
                continue
            bb = box.bounding_box()
            if bb:
                cx, cy = bb["x"] + bb["width"]/2, bb["y"] + bb["height"]/2
                if bbox["x"] <= cx <= bbox["x"] + bbox["width"] and \
                   bbox["y"] <= cy <= bbox["y"] + bbox["height"]:
                    found += 1
    except:
        pass
    return found > 0, found


def toggle_answer_display(page, activate=True):
    btn = page.locator("#AnswerDisplayButton")
    try:
        btn.wait_for(state="visible", timeout=5000)
        cls = btn.get_attribute("class") or ""
        is_active = "disp" in cls
        if activate and not is_active:
            btn.click()
            page.wait_for_timeout(800)
        elif not activate and is_active:
            btn.click()
            page.wait_for_timeout(800)
        return True
    except:
        return False


def extract_answers_from_dom(page):
    return page.evaluate("""
        () => {
            const items = document.querySelectorAll('div.answer-item');
            const results = [];
            items.forEach((el, idx) => {
                const text = (el.innerText || el.textContent || '').trim();
                if (!text) return;
                const rect = el.getBoundingClientRect();
                results.push({
                    index: idx,
                    text: text,
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    visible: rect.width > 0 && rect.height > 0
                });
            });
            return results;
        }
    """)


def take_screenshot_single_page(page, bbox, filepath):
    """
    Toma screenshot de UNA sola pagina.
    Ya NO recorta la columna izquierda: el rojo se borra en preprocesamiento.
    Solo recorta el footer de copyright.
    """
    clip = {
        "x": bbox["x"],
        "y": bbox["y"],
        "width": bbox["width"],
        "height": bbox["height"] - CROP_BOTTOM_PIXELS
    }
    page.screenshot(path=filepath, clip=clip)
    return filepath


def build_respuestas_by_side(answers, visible_pages):
    if len(visible_pages) < 2:
        return {"a": {item["index"]+1: item["text"] for item in answers}, "b": {}}

    bboxes = [v["bbox"] for v in visible_pages[:2]]
    mid_x = (bboxes[0]["x"] + bboxes[0]["width"] + bboxes[1]["x"]) / 2

    resp_a, resp_b = {}, {}
    for item in answers:
        idx = item["index"] + 1
        if item["x"] < mid_x:
            resp_a[idx] = item["text"]
        else:
            resp_b[idx] = item["text"]

    return {"a": resp_a, "b": resp_b}


def process_pair(page, save_dir, level, page_num, do_ocr):
    print(f"\n  === PAR ({level}{page_num}) ===")

    _, visible = get_page_fingerprint(page)
    if not visible:
        print("     ! Sin paginas visibles")
        return None

    # [1] Extraer respuestas del DOM
    print("     [1] Extrayendo respuestas correctas...")
    toggle_answer_display(page, activate=True)
    page.wait_for_timeout(500)

    answers = extract_answers_from_dom(page)
    resp_by_side = build_respuestas_by_side(answers, visible)

    print(f"         Pagina A: {len(resp_by_side['a'])} respuestas")
    print(f"         Pagina B: {len(resp_by_side['b'])} respuestas")

    # [2] Desactivar respuestas y tomar screenshots
    print("     [2] Tomando screenshots (sin respuestas visibles)...")
    toggle_answer_display(page, activate=False)
    page.wait_for_timeout(500)

    screenshots = []
    for idx, v in enumerate(visible[:2]):
        has_cb, _ = check_page_has_checkboxes(page, v["bbox"])
        if not has_cb:
            continue

        if len(visible) >= 2:
            is_left = v["bbox"]["x"] < visible[1]["bbox"]["x"] if idx == 0 else False
        else:
            is_left = True
        side = "a" if is_left else "b"

        filename = f"{level}{page_num}{side}.png"
        filepath = os.path.join(save_dir, filename)

        take_screenshot_single_page(page, v["bbox"], filepath)
        print(f"         [SS] {filename}")

        screenshots.append({
            "filepath": filepath,
            "side": side,
            "label": f"{level}{page_num}{side}",
            "respuestas_correctas": resp_by_side[side]
        })

    # [3] OCR + Validacion
    validations = []

    if do_ocr and MATHPIX_APP_ID and MATHPIX_APP_KEY:
        print("     [3] Ejecutando OCR (con preprocesamiento)...")

        for ss in screenshots:
            print(f"         Procesando {ss['label']}...")

            ocr_result = ocr_image(ss["filepath"])

            if "error" in ocr_result:
                print(f"         ERROR: {ocr_result['error']}")
                continue

            confidence = ocr_result.get("confidence", 0)
            ocr_text = ocr_result.get("text", "")
            print(f"         Confianza: {confidence:.1%}")

            expected_nums = list(ss["respuestas_correctas"].keys())
            ejercicios = parse_kumon_exercises(ocr_text, expected_numbers=expected_nums)

            if ejercicios:
                resultados = comparar_respuestas(ejercicios, ss["respuestas_correctas"])
                correctos, total, fuzzy = mostrar_validacion(resultados, f"Validacion {ss['label']}")

                validations.append({
                    "label": ss["label"],
                    "correctos": correctos,
                    "total": total,
                    "fuzzy": fuzzy,
                    "resultados": resultados,
                    "ocr_raw": ocr_text
                })
            else:
                print(f"         ! No se detectaron ejercicios")
                print(f"            OCR raw: {ocr_text[:150]}...")

    return {
        "page_num": page_num,
        "screenshots": screenshots,
        "respuestas": resp_by_side,
        "validations": validations
    }


def process_set(page, save_dir, level, start_page, do_ocr):
    results = []
    page_num = start_page
    last_fp = None

    while True:
        page.wait_for_timeout(500)

        fp, _ = get_page_fingerprint(page)
        if fp and fp == last_fp:
            print("  Contenido repetido -> FIN")
            break
        last_fp = fp

        result = process_pair(page, save_dir, level, page_num, do_ocr)
        if result:
            results.append(result)

        page_num += 1

        if not go_next_page(page):
            print("  No se pudo avanzar -> FIN")
            break

        page.wait_for_timeout(800)

    return results


def complete_marking(page):
    print("  Finalizando calificacion...")

    result = page.evaluate("""
        () => {
            const btn = document.querySelector('#EndScoringButton');
            if (!btn) return 'not_found';
            btn.click();
            return 'clicked';
        }
    """)

    if result == 'clicked':
        page.wait_for_timeout(2000)
        page.evaluate("""
            () => {
                const btns = [...document.querySelectorAll('button, div, span, a')];
                const confirm = btns.find(b => {
                    const t = (b.innerText || '').toLowerCase().trim();
                    return ['ok', 'yes', 'confirm', 'aceptar', 'si'].includes(t);
                });
                if (confirm) confirm.click();
            }
        """)
        page.wait_for_timeout(1500)
        return True

    return False


def mark_checkboxes(page, mode="triangle"):
    boxes = page.locator(".mark-box")
    count = 0

    try:
        for i in range(boxes.count()):
            box = boxes.nth(i)
            if not box.is_visible():
                continue

            type_div = box.locator(".mark-box-type")

            for _ in range(3):
                cls = type_div.get_attribute("class") or ""
                if mode in cls:
                    count += 1
                    break
                box.click(timeout=1000)
                time.sleep(0.15)
    except:
        pass

    return count


def run_set_workflow(page):
    level_input = input("\nNivel y pagina inicial (ej: A109, B5): ").strip()
    level, start_page = parse_level_page(level_input)

    if not level:
        print("Formato invalido")
        return

    save_dir = os.path.join(BASE_DIR, level)
    ensure_folder(save_dir)

    do_ocr = False
    if MATHPIX_APP_ID and MATHPIX_APP_KEY:
        choice = input("  Activar OCR? (s/n) [s]: ").strip().lower()
        do_ocr = choice in ("", "s", "si", "y")
    else:
        print("  ! OCR desactivado (sin credenciales)")

    print("\n  Activando vista doble...")
    activate_double_view(page)

    print(f"\n{'='*60}")
    print("  FASE 1: EXTRACCION + SCREENSHOTS")
    print('='*60)

    results = process_set(page, save_dir, level, start_page, do_ocr)

    total_screenshots = sum(len(r["screenshots"]) for r in results)
    print(f"\n  {total_screenshots} screenshots guardados")

    if do_ocr:
        total_correct = sum(v["correctos"] for r in results for v in r["validations"])
        total_fuzzy = sum(v.get("fuzzy", 0) for r in results for v in r["validations"])
        total_ex = sum(v["total"] for r in results for v in r["validations"])
        if total_ex > 0:
            efectivos = total_correct + total_fuzzy
            print(f"  Validacion: {total_correct} correctos", end="")
            if total_fuzzy > 0:
                print(f" + {total_fuzzy} revisar", end="")
            print(f" / {total_ex} ({100*efectivos/total_ex:.1f}%)")

    json_path = os.path.join(save_dir, f"{level}_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"level": level, "results": results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Datos guardados: {json_path}")

    print("\n  Regresando al inicio...")
    for _ in range(len(results)):
        try:
            smart_click(page.locator("button.up.pager-button"), "up")
            page.wait_for_timeout(500)
        except:
            break

    print(f"\n{'='*60}")
    print("  FASE 2: MARCAR CHECKBOXES")
    print('='*60)

    print("\n  Como marcar?")
    print("    [c] Todas CORRECTAS")
    print("    [i] Todas INCORRECTAS")
    print("    [s] Saltar")

    choice = input("  -> ").strip().lower()

    if choice in ("c", ""):
        mode = "triangle"
    elif choice == "i":
        mode = "check"
    else:
        mode = None

    if mode:
        pair_count = 0
        last_fp = None

        while pair_count < len(results):
            fp, _ = get_page_fingerprint(page)
            if fp == last_fp:
                break
            last_fp = fp

            marked = mark_checkboxes(page, mode)
            print(f"    Par {pair_count+1}: {marked} marcados")

            pair_count += 1
            if pair_count < len(results):
                go_next_page(page)
                page.wait_for_timeout(500)

        save = input("\n  Guardar calificacion? (s/n) [s]: ").strip().lower()
        if save in ("", "s", "si", "y"):
            complete_marking(page)
            print("  Calificacion guardada")

    print(f"\n{'='*60}")
    print(f"  Set {level} completado")
    print(f"  {os.path.abspath(save_dir)}")
    print('='*60)


def main():
    print("=" * 60)
    print("  KUMON GRADING ASSISTANT v4")
    print("=" * 60)

    if MATHPIX_APP_ID and MATHPIX_APP_KEY:
        print("  Mathpix configurado")
    else:
        print("  ! Sin Mathpix (configura MATHPIX_APP_ID/KEY)")

    print(f"  Preprocesamiento: rojo={'ON' if PREPROCESS_STRIP_RED else 'OFF'}, "
          f"upscale={PREPROCESS_UPSCALE}x, contraste={PREPROCESS_CONTRAST}")

    ensure_folder(BASE_DIR)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto("https://class-navi.digital.kumon.com/mx/index.html")

        print("\n1) Logueate en Kumon")
        print("2) Abre Instructor Marking")
        print("3) Navega al primer par del set")
        input("\nPresiona ENTER cuando estes listo...")

        try:
            page.add_style_tag(content="::-webkit-scrollbar { display: none !important; }")
        except:
            pass

        while True:
            run_set_workflow(page)

            choice = input("\nContinuar? [s]i / [n]o: ").strip().lower()
            if choice in ("n", "no", "q"):
                break

            print("\nNavega al siguiente set...")
            input("ENTER cuando estes listo...")

        print("\nHasta luego!")
        browser.close()


if __name__ == "__main__":
    main()