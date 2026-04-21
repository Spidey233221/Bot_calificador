from playwright.sync_api import sync_playwright
import anthropic
import os
import time
import re
import json
import base64
from PIL import Image, ImageFilter
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════
BASE_DIR = "base_data"

# Haiku


# Preproceso para Haiku
CROP_BOTTOM_PIXELS = 40
HEADER_MARGIN_PX = 15
TARGET_WIDTH = 1200        # Reducir imagen para ahorrar tokens
SEPARATOR_PX = 20          # Separador entre páginas A y B

# Prompt para Haiku
HAIKU_PROMPT = """Extract Kumon exercises from Page A (left) and Page B (right). 
Priority: HANDWRITTEN pencil results below the horizontal lines.

Output ONLY valid JSON, no markdown:
{"h":"PAGE_ID","e":[[NUM,"OP","RESP"],...]}

LOGIC:
1. READING ORDER: Top-to-bottom within a column, then move to the next column to the right. Page A finishes before Page B starts.
2. CONTINUITY: The last exercise of a column is followed by the top exercise of the next one. Maintain sequential numbering (1, 2, 3...) based on this flow.
3. IDENTIFICATION: Ignore all red ink and small printed keys. Only extract the messy pencil strokes.
4. EMPTY FIELDS: If a RESP is missing/illegible, use "?". If an OP is not visible, use "". 
5. FORMAT: Keep the [NUM, "OP", "RESP"] structure for every detected exercise slot, even if fields are empty."""


# ════════════════════════════════════════════════════════════════════════════
#  PREPROCESO PARA HAIKU
# ════════════════════════════════════════════════════════════════════════════

def remove_red_ink(pil_rgb_image, red_threshold=130, dominance=25, dilate=5):
    """Elimina todo pixel rojo/naranja de la imagen reemplazandolo por blanco."""
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

def preprocess_for_haiku(page, visible_pages, save_path, answers=None):
    """
    Preproceso ligero para Claude Haiku:
    1. Screenshot A: recorta header.
    2. Screenshot B: sin recorte extra.
    3. Combinar horizontalmente con separador pequeno.
    4. Reducir a TARGET_WIDTH px de ancho.
    
    Recibe answers ya extraídas del DOM (para no activar/desactivar
    respuestas de nuevo — ya se hizo en process_pair).
    Las respuestas DEBEN estar desactivadas antes de llamar esta función.
    """
    if len(visible_pages) < 2:
        print("     ! Menos de 2 paginas")
        return None

    bbox_a = visible_pages[0]["bbox"]
    bbox_b = visible_pages[1]["bbox"]

    # Calcular top de A usando answers ya extraídas
    if answers:
        a_answers = [a for a in answers
                     if bbox_a["x"] <= a["x"] <= bbox_a["x"] + bbox_a["width"]]
        if a_answers:
            top_a = max(min(a["y"] for a in a_answers) - HEADER_MARGIN_PX, bbox_a["y"])
        else:
            top_a = bbox_a["y"]
    else:
        top_a = bbox_a["y"]

    # Screenshot A (sin header, sin footer) — respuestas YA desactivadas
    clip_a = {
        "x": bbox_a["x"], "y": top_a,
        "width": bbox_a["width"],
        "height": (bbox_a["y"] + bbox_a["height"] - CROP_BOTTOM_PIXELS) - top_a
    }
    page.screenshot(path="_ha.png", clip=clip_a)

    # Screenshot B (sin footer)
    clip_b = {
        "x": bbox_b["x"], "y": bbox_b["y"],
        "width": bbox_b["width"],
        "height": bbox_b["height"] - CROP_BOTTOM_PIXELS
    }
    page.screenshot(path="_hb.png", clip=clip_b)

    # Combinar en RGB (con rojo ya eliminado)
    img_a = Image.open("_ha.png").convert("RGB")
    img_b = Image.open("_hb.png").convert("RGB")

    # Borrar todo lo rojo/naranja (X, cuadros, numeros del sistema)
    img_a = remove_red_ink(img_a)
    img_b = remove_red_ink(img_b)

    arr_a = np.array(img_a)
    arr_b = np.array(img_b)

    max_h = max(arr_a.shape[0], arr_b.shape[0])
    if arr_a.shape[0] < max_h:
        arr_a = np.vstack([arr_a, np.ones((max_h - arr_a.shape[0], arr_a.shape[1], 3), dtype=np.uint8) * 255])
    if arr_b.shape[0] < max_h:
        arr_b = np.vstack([arr_b, np.ones((max_h - arr_b.shape[0], arr_b.shape[1], 3), dtype=np.uint8) * 255])

    sep = np.ones((max_h, SEPARATOR_PX, 3), dtype=np.uint8) * 255
    combined = Image.fromarray(np.hstack([arr_a, sep, arr_b]))

    # Reducir al tamano objetivo
    w, h = combined.size
    if w > TARGET_WIDTH:
        ratio = TARGET_WIDTH / w
        combined = combined.resize((TARGET_WIDTH, int(h * ratio)), Image.LANCZOS)

    combined.save(save_path, "PNG", optimize=True)

    # Cleanup
    for f in ["_ha.png", "_hb.png"]:
        try: os.remove(f)
        except: pass

    print(f"         Imagen: {combined.size[0]}x{combined.size[1]} "
          f"({os.path.getsize(save_path) // 1024}KB)")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  OCR CON HAIKU
# ════════════════════════════════════════════════════════════════════════════

def ocr_with_haiku(image_path):
    """
    Manda imagen a Claude Haiku y recibe JSON con ejercicios.
    Devuelve dict: {"h": "hoja", "e": [[num, "op", "resp"], ...]}
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    try:
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type,
                                "data": img_data}},
                    {"type": "text", "text": HAIKU_PROMPT}
                ]
            }]
        )
    except Exception as e:
        return {"error": str(e)}

    # Extraer texto
    text = "".join(b.text for b in message.content if b.type == "text")

    # Info de uso
    usage = message.usage
    input_t = usage.input_tokens
    output_t = usage.output_tokens
    cost = input_t * 0.000001 + output_t * 0.000005
    print(f"         Tokens: {input_t} in + {output_t} out = ${cost:.4f}")

    # Parsear JSON — Haiku a veces devuelve múltiples JSONs o texto extra
    clean = text.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    # Intentar parsear directamente
    try:
        parsed = json.loads(clean)
        return parsed
    except json.JSONDecodeError:
        pass

    # Si hay múltiples JSONs, buscar todos los {...} y combinar sus ejercicios
    json_objects = []
    depth = 0
    start = -1
    for i, ch in enumerate(clean):
        if ch == '{' and depth == 0:
            start = i
        if ch == '{':
            depth += 1
        if ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(clean[start:i+1])
                    json_objects.append(obj)
                except:
                    pass
                start = -1

    if json_objects:
        # Combinar todos los ejercicios en un solo resultado
        combined = {"h": json_objects[0].get("h", "?"), "e": []}
        for obj in json_objects:
            combined["e"].extend(obj.get("e", []))
        # Re-numerar si hay duplicados (ej: ambos empiezan en 1)
        if combined["e"]:
            nums = [e[0] for e in combined["e"]]
            if len(set(nums)) < len(nums):
                # Hay duplicados — re-numerar secuencialmente
                for i, e in enumerate(combined["e"]):
                    e[0] = i + 1
        return combined

    print(f"         ! JSON invalido: {clean[:300]}")
    return {"error": "JSON invalido", "raw": clean}


def haiku_to_ejercicios(haiku_result, resp_by_side):
    """
    Convierte el JSON de Haiku al formato interno de ejercicios.
    
    Haiku devuelve: {"h": "B21a", "e": [[1, "15+5", "30"], [2, "25+5", "36"], ...]}
    Necesitamos:    {1: {"expresion": "15 + 5", "resultado_alumno": "30"}, ...}
    
    Usa resp_by_side para saber los números de ejercicio esperados.
    """
    ejercicios = {}

    if "error" in haiku_result:
        return ejercicios

    haiku_exercises = haiku_result.get("e", [])

    for entry in haiku_exercises:
        if len(entry) < 3:
            continue
        num = entry[0]
        op = str(entry[1])
        resp = str(entry[2]) if entry[2] is not None else None

        # Formatear expresión
        op_formatted = op.replace("+", " + ").replace("-", " - ").replace("*", " * ")
        # Limpiar espacios dobles
        op_formatted = re.sub(r'\s+', ' ', op_formatted).strip()

        ejercicios[num] = {
            "expresion": op_formatted,
            "resultado_alumno": resp if resp and resp != "?" else None,
            "confianza": "alta"
        }

    # Marcar no encontrados
    all_expected = set()
    for side_resp in resp_by_side.values():
        all_expected.update(side_resp.keys())

    for n in all_expected:
        if n not in ejercicios:
            ejercicios[n] = {
                "expresion": "?",
                "resultado_alumno": None,
                "confianza": "no_detectado"
            }

    return ejercicios


# ════════════════════════════════════════════════════════════════════════════
#  COMPARACIÓN Y VALIDACIÓN
# ════════════════════════════════════════════════════════════════════════════

# Confusiones comunes de lectura manuscrita (Haiku también puede confundir)
CONFUSIONES = {
    '1': ['7', '4', '|', 'l', 'i'],
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


def limpiar_respuesta_haiku(resp_raw):
    """
    Limpieza ligera de la respuesta que devuelve Haiku.
    Haiku generalmente devuelve números limpios, pero por si acaso:
    - Quita espacios
    - Quita caracteres no numéricos
    - Devuelve el número limpio o "" si ilegible
    """
    if not resp_raw or resp_raw == "?":
        return ""
    
    text = str(resp_raw).strip()
    
    # Si ya es número, devolverlo
    if text.isdigit():
        return text
    
    # Quitar espacios entre dígitos ("4 1" -> "41")
    no_spaces = text.replace(" ", "")
    if no_spaces.isdigit():
        return no_spaces
    
    # Extraer solo dígitos
    only_digits = re.sub(r'[^0-9]', '', text)
    if only_digits:
        return only_digits
    
    return ""


def es_confusion_probable(alumno_str, correcto_str):
    """
    Detecta si la diferencia entre alumno y correcto es una confusión
    visual común en manuscrito (ej: 9↔4, 1↔7, 3↔8).
    """
    if not alumno_str or not correcto_str or alumno_str == correcto_str:
        return False, None

    # Longitudes diferentes: dígito extra o faltante
    if len(alumno_str) != len(correcto_str):
        if abs(len(alumno_str) - len(correcto_str)) == 1:
            if correcto_str in alumno_str or alumno_str in correcto_str:
                return True, "digito extra/faltante"
        return False, None

    # Misma longitud: comparar dígito por dígito
    diferencias = 0
    notas = []
    for a, c in zip(alumno_str, correcto_str):
        if a != c:
            diferencias += 1
            if c in CONFUSIONES and a in CONFUSIONES[c]:
                notas.append(f"{c}->{a}")
            elif a in CONFUSIONES and c in CONFUSIONES[a]:
                notas.append(f"{c}->{a}")

    if diferencias == 1 and notas:
        return True, notas[0]
    if diferencias <= 2 and len(notas) == diferencias:
        return True, ", ".join(notas)
    return False, None


def comparar_respuestas(ejercicios_ocr, respuestas_correctas):
    """
    Compara respuestas del alumno (de Haiku) vs correctas (del DOM).
    Incluye detección de confusiones visuales comunes.
    """
    resultados = []
    for num in sorted(set(ejercicios_ocr.keys()) | set(respuestas_correctas.keys())):
        ej = ejercicios_ocr.get(num, {})
        expresion = ej.get("expresion", "?")
        alumno = ej.get("resultado_alumno")
        correcto = respuestas_correctas.get(num)

        alumno_str = limpiar_respuesta_haiku(alumno) if alumno else ""
        correcto_str = str(correcto).strip() if correcto else ""

        if not alumno_str or not correcto_str:
            if not alumno_str and correcto_str:
                es_correcto = False
                nota = "ilegible"
            else:
                es_correcto = None
                nota = None
        elif alumno_str == correcto_str:
            es_correcto = True
            nota = None
        else:
            # No coincide → verificar si es confusión visual
            es_confusion, nota_confusion = es_confusion_probable(alumno_str, correcto_str)
            if es_confusion:
                es_correcto = False
                nota = f"revisar ({nota_confusion})"
            else:
                es_correcto = False
                nota = None

        resultados.append({
            "numero": num,
            "expresion": expresion,
            "alumno": alumno_str if alumno_str else "-",
            "correcto": correcto_str if correcto_str else "-",
            "es_correcto": es_correcto,
            "nota": nota,
        })
    return resultados


def mostrar_validacion(resultados, titulo=""):
    if titulo:
        print(f"\n  [{titulo}]")
    print("  " + "-" * 75)

    correctos, incorrectos = 0, 0
    for r in resultados:
        if r["es_correcto"] is None:
            estado = "?"
        elif r["es_correcto"]:
            estado = "OK"
            correctos += 1
        elif r.get("nota") == "ilegible":
            estado = "X (ilegible)"
            incorrectos += 1
        elif r.get("nota") and "revisar" in r["nota"]:
            estado = f"X {r['nota']}"
            incorrectos += 1
        else:
            estado = "X"
            incorrectos += 1

        expr = r["expresion"][:15].ljust(15)
        print(
            f"  Ej ({r['numero']:2d}): {expr} | "
            f"Alumno: {str(r['alumno']):6s} | "
            f"Correcto: {str(r['correcto']):6s} {estado}"
        )

    print("  " + "-" * 75)
    total = correctos + incorrectos
    if total > 0:
        print(f"  Resultado: {correctos} correctos + {incorrectos} incorrectos "
              f"= {correctos}/{total} ({100*correctos/total:.1f}%)")

    return correctos, total, 0


# ════════════════════════════════════════════════════════════════════════════
#  PLAYWRIGHT (navegación y DOM)
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
                    index: idx, text: text,
                    x: Math.round(rect.x), y: Math.round(rect.y),
                    width: Math.round(rect.width), height: Math.round(rect.height),
                    visible: rect.width > 0 && rect.height > 0
                });
            });
            return results;
        }
    """)


def build_respuestas_by_side(answers, visible_pages):
    """
    Separa respuestas por lado (a=izquierda, b=derecha) y las numera
    en orden: columna 1 de arriba a abajo, columna 2 de arriba a abajo.
    
    Detecta automáticamente si hay 1 o 2 columnas dentro de cada página
    usando la coordenada X de los .answer-item. Si hay un salto grande
    en X, son 2 columnas.
    
    La numeración es CONTINUA: A empieza en 1, B continúa donde A termina.
    """
    resp_a, resp_b = {}, {}

    if len(visible_pages) < 2:
        for i, item in enumerate(answers):
            resp_a[i+1] = item["text"]
        return {"a": resp_a, "b": {}}

    bbox_a = visible_pages[0]["bbox"]
    bbox_b = visible_pages[1]["bbox"]

    def inside(item, bbox):
        return (bbox["x"] <= item["x"] <= bbox["x"] + bbox["width"] and
                bbox["y"] <= item["y"] <= bbox["y"] + bbox["height"])

    def order_by_columns(items, bbox):
        """
        Ordena items por columnas: detecta si hay 1 o 2 columnas
        y devuelve la lista en orden col1-arriba-abajo, col2-arriba-abajo.
        """
        if not items:
            return []
        
        # Detectar columnas por agrupación de X
        xs = sorted(set(item["x"] for item in items))
        
        if len(xs) <= 1:
            # Una sola columna
            return sorted(items, key=lambda x: x["y"])
        
        # Calcular punto medio entre columnas
        # Buscar el salto más grande en X
        max_gap = 0
        split_x = xs[0]
        for i in range(len(xs) - 1):
            gap = xs[i+1] - xs[i]
            if gap > max_gap:
                max_gap = gap
                split_x = (xs[i] + xs[i+1]) / 2
        
        # Si el salto es significativo (>30% del ancho de página), hay 2 columnas
        page_width = bbox["width"]
        if max_gap > page_width * 0.2:
            col1 = sorted([i for i in items if i["x"] < split_x], key=lambda x: x["y"])
            col2 = sorted([i for i in items if i["x"] >= split_x], key=lambda x: x["y"])
            return col1 + col2
        else:
            # Una columna (o items muy juntos en X)
            return sorted(items, key=lambda x: x["y"])

    a_items = [i for i in answers if inside(i, bbox_a)]
    b_items = [i for i in answers if inside(i, bbox_b)]

    a_ordered = order_by_columns(a_items, bbox_a)
    b_ordered = order_by_columns(b_items, bbox_b)

    # A empieza en 1
    for i, item in enumerate(a_ordered):
        resp_a[i + 1] = item["text"]

    # B continúa donde A terminó
    b_start = len(a_ordered) + 1
    for i, item in enumerate(b_ordered):
        resp_b[b_start + i] = item["text"]

    return {"a": resp_a, "b": resp_b}


# ════════════════════════════════════════════════════════════════════════════
#  PROCESO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

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

    toggle_answer_display(page, activate=False)
    page.wait_for_timeout(300)

    print(f"         Pagina A: {len(resp_by_side['a'])} respuestas")
    print(f"         Pagina B: {len(resp_by_side['b'])} respuestas")

    # [2] Preprocesar imagen para Haiku (respuestas ya desactivadas)
    print("     [2] Preparando imagen para Haiku...")
    combined_path = os.path.join(save_dir, f"{level}{page_num}_combined.png")
    img_path = preprocess_for_haiku(page, visible, combined_path, answers=answers)

    if not img_path:
        return {"page_num": page_num, "respuestas": resp_by_side, "validations": []}

    # [3] OCR con Haiku
    validations = []
    if do_ocr:
        print("     [3] Llamando a Claude Haiku...")
        haiku_result = ocr_with_haiku(img_path)

        if "error" in haiku_result:
            print(f"         ERROR: {haiku_result['error']}")
        else:
            hoja = haiku_result.get("h", "?")
            ejercicios_raw = haiku_result.get("e", [])
            print(f"         Hoja: {hoja}  |  Ejercicios: {len(ejercicios_raw)}")

            # Guardar cache
            cache_path = os.path.join(save_dir, f"{level}{page_num}_ocr.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(haiku_result, f, indent=2, ensure_ascii=False)
            print(f"         [CACHE] {os.path.basename(cache_path)}")

            # Convertir a formato interno
            ejercicios = haiku_to_ejercicios(haiku_result, resp_by_side)

            # Separar por lado
            nums_a = set(resp_by_side["a"].keys())
            nums_b = set(resp_by_side["b"].keys())

            for label, nums, resp_correctas in [
                (f"{level}{page_num}a", nums_a, resp_by_side["a"]),
                (f"{level}{page_num}b", nums_b, resp_by_side["b"]),
            ]:
                ej_side = {n: v for n, v in ejercicios.items() if n in nums}
                if ej_side:
                    resultados = comparar_respuestas(ej_side, resp_correctas)
                    correctos, total, _ = mostrar_validacion(resultados, f"Validacion {label}")
                    validations.append({
                        "label": label,
                        "correctos": correctos,
                        "total": total,
                        "resultados": resultados,
                        "haiku_raw": haiku_result
                    })

    return {
        "page_num": page_num,
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


# ════════════════════════════════════════════════════════════════════════════
#  MARKING (checkboxes)
# ════════════════════════════════════════════════════════════════════════════

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


def get_all_markboxes_in_viewport(page, visible_pages):
    boxes = page.locator(".mark-box")
    count = boxes.count()
    if len(visible_pages) < 2:
        return {"a": [], "b": []}

    bbox_a = visible_pages[0]["bbox"]
    bbox_b = visible_pages[1]["bbox"]
    a_boxes, b_boxes = [], []

    for i in range(count):
        box = boxes.nth(i)
        try:
            bb = box.bounding_box()
            if not bb or bb["width"] < 5: continue
            cx = bb["x"] + bb["width"] / 2
            cy = bb["y"] + bb["height"] / 2
            if (bbox_a["x"] <= cx <= bbox_a["x"] + bbox_a["width"] and
                bbox_a["y"] <= cy <= bbox_a["y"] + bbox_a["height"]):
                a_boxes.append({"locator": box, "y": bb["y"], "x": bb["x"]})
            elif (bbox_b["x"] <= cx <= bbox_b["x"] + bbox_b["width"] and
                  bbox_b["y"] <= cy <= bbox_b["y"] + bbox_b["height"]):
                b_boxes.append({"locator": box, "y": bb["y"], "x": bb["x"]})
        except: pass

    a_boxes.sort(key=lambda b: b["y"])
    b_boxes.sort(key=lambda b: b["y"])
    return {"a": a_boxes, "b": b_boxes}


def click_to_state(box_locator, target_state, max_clicks=4):
    type_div = box_locator.locator(".mark-box-type")
    for _ in range(max_clicks):
        try:
            cls = type_div.get_attribute("class") or ""
            if target_state in cls:
                return True
            box_locator.click(timeout=1000)
            time.sleep(0.2)
        except:
            return False
    return False


def mark_checkboxes_by_result(page, visible_pages, validations, resp_by_side):
    results_by_num = {}
    for v in validations:
        for r in v.get("resultados", []):
            results_by_num[r["numero"]] = r

    if not results_by_num:
        print("         ! Sin resultados para marcar")
        return 0

    toggle_answer_display(page, activate=True)
    page.wait_for_timeout(500)
    answers = extract_answers_from_dom(page)
    toggle_answer_display(page, activate=False)
    page.wait_for_timeout(300)

    cb = get_all_markboxes_in_viewport(page, visible_pages)
    if len(visible_pages) < 2:
        return 0

    bbox_a = visible_pages[0]["bbox"]
    bbox_b = visible_pages[1]["bbox"]

    def inside_bbox(item, bbox):
        return (bbox["x"] <= item["x"] <= bbox["x"] + bbox["width"] and
                bbox["y"] <= item["y"] <= bbox["y"] + bbox["height"])

    a_items = sorted([a for a in answers if inside_bbox(a, bbox_a)], key=lambda x: x["y"])
    b_items = sorted([a for a in answers if inside_bbox(a, bbox_b)], key=lambda x: x["y"])
    a_nums = sorted(resp_by_side.get("a", {}).keys())
    b_nums = sorted(resp_by_side.get("b", {}).keys())

    a_answers = [{"y": item["y"], "num": a_nums[i]} for i, item in enumerate(a_items) if i < len(a_nums)]
    b_answers = [{"y": item["y"], "num": b_nums[i]} for i, item in enumerate(b_items) if i < len(b_nums)]

    total_marked = 0
    for side, side_boxes, side_answers in [("a", cb["a"], a_answers), ("b", cb["b"], b_answers)]:
        if not side_boxes or not side_answers: continue
        for box_info in side_boxes:
            best = min(side_answers, key=lambda a: abs(box_info["y"] - a["y"]), default=None)
            if not best or abs(box_info["y"] - best["y"]) > 50: continue

            result = results_by_num.get(best["num"])
            if not result: continue

            if result.get("es_correcto"):
                total_marked += 1
                continue

            try:
                if click_to_state(box_info["locator"], "check"):
                    total_marked += 1
                    print(f"         Ej ({best['num']:2d}): X")
                else:
                    print(f"         ! Ej ({best['num']:2d}): no se pudo marcar")
            except Exception as e:
                print(f"         ! Ej ({best['num']:2d}): error: {e}")

    return total_marked


def mark_checkboxes(page, mode="triangle"):
    boxes = page.locator(".mark-box")
    count = 0
    try:
        for i in range(boxes.count()):
            box = boxes.nth(i)
            if not box.is_visible(): continue
            type_div = box.locator(".mark-box-type")
            for _ in range(3):
                cls = type_div.get_attribute("class") or ""
                if mode in cls: count += 1; break
                box.click(timeout=1000)
                time.sleep(0.15)
    except: pass
    return count


# ════════════════════════════════════════════════════════════════════════════
#  WORKFLOW
# ════════════════════════════════════════════════════════════════════════════

def run_set_workflow(page):
    level_input = input("\nNivel y pagina inicial (ej: A109, B5): ").strip()
    level, start_page = parse_level_page(level_input)
    if not level:
        print("Formato invalido")
        return

    save_dir = os.path.join(BASE_DIR, level)
    ensure_folder(save_dir)

    choice = input("  Activar OCR con Haiku? (s/n) [s]: ").strip().lower()
    do_ocr = choice in ("", "s", "si", "y")

    print("\n  Activando vista doble...")
    activate_double_view(page)

    print(f"\n{'='*60}\n  FASE 1: EXTRACCION + OCR (Haiku)\n{'='*60}")
    results = process_set(page, save_dir, level, start_page, do_ocr)

    if do_ocr:
        total_correct = sum(v["correctos"] for r in results for v in r.get("validations", []))
        total_ex = sum(v["total"] for r in results for v in r.get("validations", []))
        if total_ex > 0:
            print(f"\n  TOTAL: {total_correct}/{total_ex} ({100*total_correct/total_ex:.1f}%)")

    json_path = os.path.join(save_dir, f"{level}_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"level": level, "results": results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Datos guardados: {json_path}")

    # Regresar al inicio
    print("\n  Regresando al inicio...")
    for _ in range(len(results)):
        try:
            smart_click(page.locator("button.up.pager-button"), "up")
            page.wait_for_timeout(500)
        except: break

    # FASE 2: Marcar checkboxes
    print(f"\n{'='*60}\n  FASE 2: MARCAR CHECKBOXES\n{'='*60}")
    print("\n  Como marcar?\n    [a] AUTOMATICO\n    [c] Todas CORRECTAS\n    [i] Todas INCORRECTAS\n    [s] Saltar")
    choice = input("  -> ").strip().lower()

    if choice in ("a", ""):
        print("\n  Marcando segun resultados...")
        pair_count = 0
        last_fp = None
        while pair_count < len(results):
            fp, visible = get_page_fingerprint(page)
            if fp == last_fp: break
            last_fp = fp
            r = results[pair_count]
            marked = mark_checkboxes_by_result(page, visible, r.get("validations", []), r.get("respuestas", {}))
            print(f"    Par {pair_count+1}: {marked} marcadas")
            pair_count += 1
            if pair_count < len(results):
                go_next_page(page)
                page.wait_for_timeout(500)

        save = input("\n  Guardar calificacion? (s/n) [s]: ").strip().lower()
        if save in ("", "s", "si", "y"):
            complete_marking(page)
            print("  Calificacion guardada")

    elif choice in ("c", "i"):
        mode = "triangle" if choice == "c" else "check"
        pair_count = 0
        last_fp = None
        while pair_count < len(results):
            fp, _ = get_page_fingerprint(page)
            if fp == last_fp: break
            last_fp = fp
            marked = mark_checkboxes(page, mode)
            print(f"    Par {pair_count+1}: {marked} marcados")
            pair_count += 1
            if pair_count < len(results):
                go_next_page(page)
                page.wait_for_timeout(500)
        save = input("\n  Guardar? (s/n) [s]: ").strip().lower()
        if save in ("", "s", "si", "y"):
            complete_marking(page)

    print(f"\n{'='*60}\n  Set {level} completado\n  {os.path.abspath(save_dir)}\n{'='*60}")


def main():
    print("=" * 60)
    print("  KUMON GRADING ASSISTANT v6 (Haiku)")
    print("=" * 60)
    print(f"  OCR: Claude Haiku 4.5")
    print(f"  Imagen: {TARGET_WIDTH}px ancho (reducida)")

    ensure_folder(BASE_DIR)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto("https://class-navi.digital.kumon.com/mx/index.html")

        print("\n1) Logueate en Kumon\n2) Abre Instructor Marking\n3) Navega al primer par")
        input("\nPresiona ENTER cuando estes listo...")

        try:
            page.add_style_tag(content="::-webkit-scrollbar { display: none !important; }")
        except: pass

        while True:
            run_set_workflow(page)
            choice = input("\nContinuar? [s]i / [n]o: ").strip().lower()
            if choice in ("n", "no", "q"): break
            print("\nNavega al siguiente set...")
            input("ENTER cuando estes listo...")

        print("\nHasta luego!")
        browser.close()


if __name__ == "__main__":
    main()