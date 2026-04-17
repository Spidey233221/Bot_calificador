from playwright.sync_api import sync_playwright
import os
import time
import re
import json
import requests
import base64
from PIL import Image, ImageFilter
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════
BASE_DIR = "base_data"

MATHPIX_APP_ID = "edtechsa_edabfb_05c62a"
MATHPIX_APP_KEY = "b3472076544729f5a892d8ecb150e7b9205ab308c337b5d3aef69ba804cf61ab"

CROP_BOTTOM_PIXELS = 40
HEADER_MARGIN_PX = 15

PREPROCESS_STRIP_RED = True
PREPROCESS_UPSCALE = 3
PREPROCESS_GAMMA = 0.75
BINARY_THRESHOLD = 200     # Umbral binario: <200 → negro, >=200 → blanco
MORPH_CLOSE_SIZE = 3       # Tamaño del filtro de cierre (0 o 1 = desactivado)

# Modo combinado: enviar A+B en UNA sola llamada a Mathpix (ahorra 50% de costo)
COMBINED_OCR_MODE = True       # True = 1 llamada por par, False = 2 llamadas
COMBINED_SEPARATOR_PX = 80     # Ancho del separador blanco entre A y B

# ════════════════════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO DE IMAGEN
# ════════════════════════════════════════════════════════════════════════════

def compact_horizontal_gaps(pil_gray_image, gap_target_px=35,
                            min_gap_to_compact=60, dark_threshold=150):
    """
    Comprime los huecos horizontales grandes dentro de cada línea de texto.

    Útil para la página B de Kumon donde los paréntesis (12), (13)...
    están muy lejos de los operandos. Mathpix interpreta ese hueco como
    estructura de tabla y el OCR colapsa. Al comprimir el hueco a ~35px,
    cada línea se ve como "(12) 2 + 6 = 8" igual que en A.

    Detecta filas con contenido, y dentro de cada fila busca la columna
    en blanco más larga y la reemplaza por un gap pequeño fijo.
    """
    arr = np.array(pil_gray_image)
    h, w = arr.shape

    # Detectar los bordes del marco negro (si existen)
    mid_y = h // 2
    col_profile = arr[max(0, mid_y-5):mid_y+5, :].mean(axis=0)
    frame_cols = np.where(col_profile < 100)[0]
    if len(frame_cols) > 0:
        left_margin = int(frame_cols[0]) + 10
        right_margin_excl = w - int(frame_cols[-1]) + 10
    else:
        left_margin = 10
        right_margin_excl = 10

    # Detectar líneas de contenido
    interior = arr[:, left_margin:w-right_margin_excl]
    dark_count = (interior < dark_threshold).sum(axis=1)
    row_has_content = dark_count > 5

    lines = []
    in_line = False
    start = 0
    for y, has in enumerate(row_has_content):
        if has and not in_line:
            start = y
            in_line = True
        elif not has and in_line:
            lines.append((start, y))
            in_line = False
    if in_line:
        lines.append((start, h))
    lines = [(s, e) for s, e in lines if (e - s) >= 25]

    if not lines:
        return pil_gray_image

    # Componer imagen nueva
    new_rows = []
    for (s, e) in lines:
        line_img = arr[s:e, left_margin:w-right_margin_excl]
        lh = e - s

        col_dark = (line_img < dark_threshold).sum(axis=0)
        content_cols = np.where(col_dark > 0)[0]
        if len(content_cols) < 2:
            new_rows.append(line_img)
            new_rows.append(np.ones((15, line_img.shape[1]), dtype=np.uint8) * 255)
            continue

        left_b = int(content_cols[0])
        right_b = int(content_cols[-1])

        # Buscar hueco más grande DENTRO del contenido
        empty_col = col_dark == 0
        max_len = 0
        max_start = 0
        cur_start = 0
        cur_len = 0
        for x in range(left_b, right_b + 1):
            if empty_col[x]:
                if cur_len == 0:
                    cur_start = x
                cur_len += 1
                if cur_len > max_len:
                    max_len = cur_len
                    max_start = cur_start
            else:
                cur_len = 0

        if max_len > min_gap_to_compact:
            left_part = line_img[:, max(left_b-5, 0):max_start]
            right_part = line_img[:, max_start+max_len:min(right_b+5, line_img.shape[1])]
            gap = np.ones((lh, gap_target_px), dtype=np.uint8) * 255
            new_line = np.hstack([left_part, gap, right_part])
        else:
            new_line = line_img[:, max(left_b-5, 0):min(right_b+5, line_img.shape[1])]

        new_rows.append(new_line)
        new_rows.append(np.ones((15, new_line.shape[1]), dtype=np.uint8) * 255)

    max_w = max(r.shape[1] for r in new_rows)
    new_rows_padded = []
    for r in new_rows:
        if r.shape[1] < max_w:
            pad = np.ones((r.shape[0], max_w - r.shape[1]), dtype=np.uint8) * 255
            r = np.hstack([r, pad])
        new_rows_padded.append(r)

    result = np.vstack(new_rows_padded)

    # Pad blanco alrededor
    pad = 30
    padded = np.ones((result.shape[0] + 2*pad, result.shape[1] + 2*pad),
                     dtype=np.uint8) * 255
    padded[pad:pad+result.shape[0], pad:pad+result.shape[1]] = result
    return Image.fromarray(padded)


def remove_red_ink(pil_rgb_image, red_threshold=130, dominance=25, dilate=5):
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


def preprocess_pil(img_rgb, target_width=None, compact_gaps=False,
                   source_width=None):
    """
    Preproceso en memoria: recibe PIL RGB, devuelve PIL L.
    Pipeline simple que ya funcionaba bien:
    1. Borrar rojo.
    2. Upscale.
    3. Grises.
    4. Compactar huecos horizontales (solo si compact_gaps=True).
    5. Gamma suave.

    SIN umbralización binaria ni cierre morfológico — eso engrosaba
    demasiado los trazos y los paréntesis se volvían manchas.
    """
    img = img_rgb

    if PREPROCESS_STRIP_RED:
        img = remove_red_ink(img)

    src_w = source_width if source_width else img.width
    if target_width and src_w > 0:
        target_final_width = target_width * PREPROCESS_UPSCALE
        scale_factor = target_final_width / src_w
    else:
        scale_factor = PREPROCESS_UPSCALE


    if scale_factor and scale_factor != 1:
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)

    img = img.convert("L")

    if compact_gaps:
        img = compact_horizontal_gaps(img)

    gamma = PREPROCESS_GAMMA
    lut = [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)]
    img = img.point(lut)

    img.save("debug.png")
    return img


def combine_pages_horizontal(img_a, img_b, separator_px=80):
    """
    Concatena dos páginas preprocesadas horizontalmente con un separador
    blanco en el medio. Esto permite enviar ambas en una sola llamada a
    Mathpix. Las páginas deben estar ya en modo L (escala de grises).

    Altura: usa la mayor de las dos, pad con blanco la más corta.
    Padding vertical pequeño arriba y abajo.
    """
    # Forzar mismo modo
    if img_a.mode != "L":
        img_a = img_a.convert("L")
    if img_b.mode != "L":
        img_b = img_b.convert("L")

    a = np.array(img_a)
    b = np.array(img_b)

    # Igualar alturas con padding blanco
    max_h = max(a.shape[0], b.shape[0])
    if a.shape[0] < max_h:
        pad = np.ones((max_h - a.shape[0], a.shape[1]), dtype=np.uint8) * 255
        a = np.vstack([a, pad])
    if b.shape[0] < max_h:
        pad = np.ones((max_h - b.shape[0], b.shape[1]), dtype=np.uint8) * 255
        b = np.vstack([b, pad])

    # Separador blanco entre páginas
    sep = np.ones((max_h, separator_px), dtype=np.uint8) * 255

    combined = np.hstack([a, sep, b])

    # Padding vertical
    pad_v = 40
    vpad = np.ones((pad_v, combined.shape[1]), dtype=np.uint8) * 255
    combined = np.vstack([vpad, combined, vpad])

    return Image.fromarray(combined)


def preprocess_image(input_path, output_path=None, target_width=None,
                     compact_gaps=False):
    """
    Wrapper file-to-file de preprocess_pil. Se mantiene para compatibilidad
    con el modo "OCR por página individual" (fallback).
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_pp{ext}"

    img = Image.open(input_path).convert("RGB")
    result = preprocess_pil(img, target_width=target_width,
                            compact_gaps=compact_gaps)
    result.save(output_path, "PNG", optimize=True)
    return output_path


# ════════════════════════════════════════════════════════════════════════════
#  OCR (Mathpix)
# ════════════════════════════════════════════════════════════════════════════

def ocr_image(image_path, target_width=None, compact_gaps=False):
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        return {"error": "Credenciales Mathpix no configuradas"}

    try:
        path_to_send = preprocess_image(image_path, target_width=target_width,
                                        compact_gaps=compact_gaps)
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


def ocr_file_direct(image_path):
    """
    Envía una imagen a Mathpix SIN preprocesamiento adicional.
    Se usa cuando la imagen ya fue preprocesada (ej: combinación A+B).
    """
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        return {"error": "Credenciales Mathpix no configuradas"}

    with open(image_path, "rb") as f:
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
    Parser robusto con inferencia completa de numeros de ejercicio.
    Pasada 1: extrae lineas con ecuaciones (con o sin numero).
    Pasada 2: asigna numeros por contexto secuencial.
    """
    if not ocr_text:
        if expected_numbers:
            return {n: {"expresion": "?", "resultado_alumno": None,
                        "confianza": "no_detectado"} for n in expected_numbers}
        return {}

    expected_set = set(expected_numbers) if expected_numbers else None

    text = ocr_text
    text = re.sub(r'\\begin\{array\}\{[^}]*\}', '', text)
    text = re.sub(r'\\end\{array\}', '', text)
    text = re.sub(r'\\\\', '\n', text)

    lines_text = text.split('\n')

    # === PASADA 1: extraer todas las lineas con info ===
    parsed_lines = []

    for line in lines_text:
        line = line.strip()
        if not line:
            continue

        # Intentar extraer numero de ejercicio
        pm = re.search(r'(?:^|\s)\(?(\d{1,2})\)[\s:]', line)
        if not pm:
            pm = re.match(r'(\d{1,2})\)\s', line)
        if not pm:
            pm = re.match(r'\((\d{1,2})\)\s', line)

        raw_num = int(pm.group(1)) if pm else None
        after = line[pm.end():] if pm else line

        # Buscar ecuacion A op B = R
        em = re.search(r'(\d+)\s*([\+\-\*x])\s*(\d+)\s*=\s*(.*)', after)
        if not em and pm:
            rest = line[pm.start() + len(pm.group(0)):]
            em = re.search(r'(\d+)\s*([\+\-\*x])\s*(\d+)\s*=\s*(.*)', rest)

        if em:
            op = em.group(2)
            if op in ('x',):
                op = '*'
            parsed_lines.append({
                "raw_num": raw_num,
                "expresion": f"{em.group(1)} {op} {em.group(3)}",
                "respuesta": em.group(4).strip() or None,
                "confianza": "alta",
            })
        elif raw_num is not None:
            eq = re.search(r'=\s*(.*)', line)
            if eq:
                parsed_lines.append({
                    "raw_num": raw_num,
                    "expresion": "?",
                    "respuesta": eq.group(1).strip() or None,
                    "confianza": "media",
                })
            else:
                parsed_lines.append({
                    "raw_num": raw_num,
                    "expresion": "?",
                    "respuesta": None,
                    "confianza": "baja",
                })
        else:
            # Sin numero — si tiene ecuacion, registrar para relleno
            eq = re.search(r'(\d+)\s*([\+\-\*x])\s*(\d+)\s*=\s*(.*)', line)
            if eq:
                op = eq.group(2)
                if op in ('x',):
                    op = '*'
                parsed_lines.append({
                    "raw_num": None,
                    "expresion": f"{eq.group(1)} {op} {eq.group(3)}",
                    "respuesta": eq.group(4).strip() or None,
                    "confianza": "media",
                })

    # === PASADA 2: asignar numeros definitivos ===
    ejercicios = {}
    last_assigned = 0

    for pl in parsed_lines:
        raw_num = pl["raw_num"]
        num = None

        if raw_num is not None:
            num = raw_num

            # Correccion secuencial
            if last_assigned > 0 and num <= last_assigned:
                for offset in [10, 20, 30]:
                    corrected = raw_num + offset
                    if expected_set:
                        if corrected in expected_set and corrected not in ejercicios:
                            num = corrected
                            break
                    elif corrected == last_assigned + 1:
                        num = corrected
                        break

                if num <= last_assigned and expected_set:
                    nxt = last_assigned + 1
                    if nxt in expected_set and nxt not in ejercicios:
                        num = nxt

            # Validar contra expected
            if expected_set and num not in expected_set:
                nxt = last_assigned + 1
                if nxt in expected_set and nxt not in ejercicios:
                    num = nxt
                else:
                    continue

        else:
            # Sin numero — inferir siguiente
            nxt = last_assigned + 1
            if expected_set:
                if nxt in expected_set and nxt not in ejercicios:
                    num = nxt
                else:
                    for candidate in sorted(expected_set):
                        if candidate > last_assigned and candidate not in ejercicios:
                            num = candidate
                            break
            else:
                num = nxt

        if num is None or num in ejercicios:
            continue

        ejercicios[num] = {
            "expresion": pl["expresion"],
            "resultado_alumno": pl["respuesta"],
            "confianza": pl["confianza"],
        }
        last_assigned = num

    # Marcar no encontrados
    if expected_numbers:
        for n in expected_numbers:
            if n not in ejercicios:
                ejercicios[n] = {
                    "expresion": "?",
                    "resultado_alumno": None,
                    "confianza": "no_detectado",
                }

    return ejercicios


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


# ════════════════════════════════════════════════════════════════════════════
#  HEURÍSTICAS LOCALES: Limpieza de OCR manuscrito
# ════════════════════════════════════════════════════════════════════════════

# Mapeo multi-candidato: cada carácter puede representar varios dígitos
CHAR_TO_DIGIT_MULTI = {
    'o': ['0'], 'O': ['0'], 'D': ['0'], 'Q': ['0'],
    'l': ['1'], 'I': ['1'], 'i': ['1'], '|': ['1'], '/': ['1'],
    'r': ['1', '2'], 'T': ['1'],
    'z': ['2'], 'Z': ['2'],
    ')': ['3'], '}': ['3'], 'E': ['3'],
    'h': ['4'], 'H': ['4'],
    's': ['1', '5'], 'S': ['1', '5'], '$': ['5'],
    'b': ['6'], 'G': ['6'],
    '?': ['7'],
    'g': ['8'], '&': ['8'], 'B': ['3', '8'],
    'q': ['9'], 'a': ['9', '0'],
}

# Palabras/patrones completos que representan números
# SOLO los que son OBVIOS y no tienen ambigüedad.
# 'n' NO se incluye porque puede ser 10, 11, o un trazo random.
WORD_TO_NUMBER = {
    'ro': '10',   # "r o" sin espacio: r=1, o=0 → claramente 10
    'lo': '10',   # "l o": l=1, o=0
    'io': '10',   # "i o": i=1, o=0
}

# Rango máximo de respuestas por nivel (para filtrar candidatos imposibles)
MAX_ANSWER_BY_LEVEL = {
    'A': 20, 'B': 200, 'C': 10000, 'D': 100000,
}


def limpiar_ocr_alumno(ocr_raw, operando1=None, operador=None, operando2=None,
                        max_answer=20):
    """
    Limpia el texto crudo del OCR de la respuesta del alumno y devuelve
    el mejor número entero posible.

    Pipeline:
    1. Quitar espacios y LaTeX.
    2. Si ya es número puro, devolverlo.
    3. Mapear palabras completas conocidas (n→10).
    4. Generar TODAS las combinaciones posibles mapeando cada carácter
       a sus candidatos de dígito.
    5. Filtrar por rango válido (0 a max_answer).
    6. Si hay un solo candidato válido, devolver ese.
    7. Si hay varios, preferir el más corto/pequeño.

    Args:
        ocr_raw: texto crudo de Mathpix para este ejercicio.
        max_answer: respuesta máxima válida para este nivel.

    Returns:
        dict con:
          - limpio: str con el número interpretado (o "" si no pudo)
          - metodo: str describiendo qué heurística lo resolvió
          - confianza: "alta" | "media" | "baja"
    """
    import itertools

    if not ocr_raw:
        return {"limpio": "", "metodo": "sin_dato", "confianza": "baja"}

    original = ocr_raw.strip()

    # Quitar LaTeX
    text = original
    text = re.sub(r'\\[a-zA-Z]+\s*', '', text)
    text = re.sub(r'[{}\[\]\\]', '', text)

    # Quitar espacios
    text_clean = text.replace(' ', '')

    # Paso 1: ya es número puro
    if text_clean.isdigit():
        return {"limpio": text_clean, "metodo": "directo", "confianza": "alta"}

    # Paso 2: palabra/patrón conocido
    if text_clean.lower() in WORD_TO_NUMBER:
        result = WORD_TO_NUMBER[text_clean.lower()]
        return {"limpio": result, "metodo": f"patron:{text_clean}->{result}",
                "confianza": "media"}

    # Paso 3: mapeo multi-candidato
    options_per_char = []
    mappings_used = False
    for ch in text_clean:
        if ch.isdigit():
            options_per_char.append([ch])
        elif ch in CHAR_TO_DIGIT_MULTI:
            options_per_char.append(CHAR_TO_DIGIT_MULTI[ch])
            mappings_used = True

    if not options_per_char:
        return {"limpio": "", "metodo": "ilegible", "confianza": "baja"}

    # Generar combinaciones (limitar a 64 para no explotar)
    total_combos = 1
    for opts in options_per_char:
        total_combos *= len(opts)
    if total_combos > 64:
        # Demasiadas combinaciones, usar solo primera opción de cada char
        single = ''.join(opts[0] for opts in options_per_char)
        if single.isdigit():
            return {"limpio": single, "metodo": "charmap_truncado",
                    "confianza": "baja"}
        return {"limpio": "", "metodo": "ilegible", "confianza": "baja"}

    candidates = set()
    for combo in itertools.product(*options_per_char):
        num_str = ''.join(combo)
        if num_str.isdigit():
            candidates.add(num_str)

    if not candidates:
        # Fallback: solo dígitos del texto original
        solo_digitos = re.sub(r'[^0-9]', '', text)
        if solo_digitos:
            return {"limpio": solo_digitos, "metodo": "solo_digitos",
                    "confianza": "baja"}
        return {"limpio": "", "metodo": "ilegible", "confianza": "baja"}

    # Filtrar por rango válido
    valid = [c for c in candidates if 0 <= int(c) <= max_answer]

    if len(valid) == 1:
        return {"limpio": valid[0],
                "metodo": f"multi_candidato:{text_clean}->{valid[0]}",
                "confianza": "media" if mappings_used else "alta"}
    elif len(valid) > 1:
        valid.sort(key=lambda x: (len(x), int(x)))
        return {"limpio": valid[0],
                "metodo": f"multi_mejor:{text_clean}->{valid[0]}",
                "confianza": "media"}

    # Ninguno en rango — devolver el candidato numérico más chico
    all_sorted = sorted(candidates, key=lambda x: int(x))
    return {"limpio": all_sorted[0],
            "metodo": f"fuera_rango:{text_clean}->{all_sorted[0]}",
            "confianza": "baja"}


def comparar_respuestas(ejercicios_ocr, respuestas_correctas):
    """
    Compara respuestas con limpieza heurística + validación fuzzy.

    Flujo por ejercicio:
    1. Limpiar el OCR crudo del alumno con heurísticas.
    2. Si el limpio == correcto → match directo.
    3. Si no, verificar si es confusión OCR conocida → match fuzzy.
    4. Si no, marcar como incorrecto (error real del alumno).
    """
    resultados = []
    for num in sorted(set(ejercicios_ocr.keys()) | set(respuestas_correctas.keys())):
        ej = ejercicios_ocr.get(num, {})
        expresion = ej.get("expresion", "?")
        alumno_raw = ej.get("resultado_alumno")
        correcto = respuestas_correctas.get(num)

        correcto_str = str(correcto).strip() if correcto else ""

        # Limpiar el OCR con heurísticas
        if alumno_raw:
            # Extraer operandos de la expresión para verificación aritmética
            expr_match = re.match(r'(\d+)\s*([\+\-])\s*(\d+)', expresion) if expresion else None
            op1 = int(expr_match.group(1)) if expr_match else None
            operador = expr_match.group(2) if expr_match else None
            op2 = int(expr_match.group(3)) if expr_match else None

            limpieza = limpiar_ocr_alumno(str(alumno_raw), op1, operador, op2)
            alumno_str = limpieza["limpio"]
            metodo_limpieza = limpieza["metodo"]
            confianza_limpieza = limpieza["confianza"]
        else:
            alumno_str = ""
            metodo_limpieza = "sin_dato"
            confianza_limpieza = "baja"

        # Comparar
        if not alumno_str or not correcto_str:
            if not alumno_str and correcto_str:
                # No se pudo leer la respuesta → marcar como incorrecto
                es_correcto = False
                es_fuzzy = False
                nota_fuzzy = "ilegible"
            else:
                es_correcto, es_fuzzy, nota_fuzzy = None, False, None
        elif alumno_str == correcto_str:
            if confianza_limpieza in ("alta", "media"):
                # Match directo o heurística razonable → correcto
                es_correcto, es_fuzzy, nota_fuzzy = True, False, None
            else:
                # Confianza baja: coincide pero la limpieza fue muy dudosa
                # → marcar como incorrecto con nota de revisar (LLM después)
                es_correcto = False
                es_fuzzy = False
                nota_fuzzy = f"revisar_llm:{metodo_limpieza}"
        else:
            # No coincide → verificar confusiones OCR conocidas
            es_fuzzy, nota_fuzzy = es_error_ocr_probable(alumno_str, correcto_str)
            es_correcto = False

        resultados.append({
            "numero": num,
            "expresion": expresion,
            "alumno_raw": str(alumno_raw) if alumno_raw else "-",
            "alumno": alumno_str if alumno_str else "-",
            "correcto": correcto_str if correcto_str else "-",
            "es_correcto": es_correcto,
            "es_fuzzy": es_fuzzy,
            "nota_fuzzy": nota_fuzzy
        })
    return resultados


def mostrar_validacion(resultados, titulo=""):
    if titulo:
        print(f"\n  [{titulo}]")
    print("  " + "-" * 85)

    correctos, incorrectos, fuzzy, sin_respuesta = 0, 0, 0, 0
    for r in resultados:
        if r["es_correcto"] is None:
            estado = "?"
            sin_respuesta += 1
        elif r["es_correcto"] and not r.get("es_fuzzy"):
            estado = "OK"
            correctos += 1
        elif r["es_correcto"] and r.get("es_fuzzy"):
            estado = f"OK* ({r['nota_fuzzy']})" if r.get("nota_fuzzy") else "OK*"
            correctos += 1
        elif r.get("es_fuzzy"):
            estado = f"X revisar ({r['nota_fuzzy']})" if r.get("nota_fuzzy") else "X revisar"
            incorrectos += 1
        elif r.get("nota_fuzzy") == "ilegible":
            estado = "X (ilegible)"
            incorrectos += 1
        elif r.get("nota_fuzzy") and r["nota_fuzzy"].startswith("revisar_llm"):
            estado = f"X ({r['nota_fuzzy']})"
            incorrectos += 1
        else:
            estado = "X"
            incorrectos += 1

        expr = r["expresion"][:12].ljust(12)
        alumno_display = r.get("alumno_raw", r["alumno"])
        limpio_display = r["alumno"]

        # Mostrar raw→limpio si son distintos
        if alumno_display != limpio_display and limpio_display != "-":
            alumno_col = f"{alumno_display}->{limpio_display}"
        else:
            alumno_col = str(alumno_display)

        print(
            f"  Ej ({r['numero']:2d}): {expr} | "
            f"Alumno: {alumno_col:14s} | "
            f"Correcto: {str(r['correcto']):4s} {estado}"
        )

    print("  " + "-" * 85)
    total = correctos + incorrectos + fuzzy
    if total > 0:
        efectivos = correctos + fuzzy
        print(f"  Resultado: {correctos} correctos", end="")
        if fuzzy > 0: print(f" + {fuzzy} probables (revisar)", end="")
        if incorrectos > 0: print(f" + {incorrectos} incorrectos", end="")
        print(f" = {efectivos}/{total} ({100*efectivos/total:.1f}%)")
        if sin_respuesta > 0:
            print(f"      {sin_respuesta} ejercicios sin respuesta detectada")

    return correctos, total, fuzzy


# ════════════════════════════════════════════════════════════════════════════
#  PLAYWRIGHT
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
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    visible: rect.width > 0 && rect.height > 0
                });
            });
            return results;
        }
    """)


def take_screenshot_single_page(page, bbox, filepath,
                                answer_cells=None, side="a"):
    """
    Screenshot con recorte contextual:
    - Página A: recorta header (título + tabla de notas + "Suma.").
    - Página B: captura completa. Los bordes negros y los huecos
      horizontales los maneja compact_horizontal_gaps() durante el
      preprocesamiento.
    - Ambas: footer de copyright fuera.
    """
    top = bbox["y"]

    if side == "a" and answer_cells:
        min_y = min(c["y"] for c in answer_cells)
        top = max(min_y - HEADER_MARGIN_PX, bbox["y"])

    clip = {
        "x": bbox["x"],
        "y": top,
        "width": bbox["width"],
        "height": (bbox["y"] + bbox["height"] - CROP_BOTTOM_PIXELS) - top
    }
    page.screenshot(path=filepath, clip=clip)

    print("width", bbox["width"])
    return {
        "filepath": filepath,
        "target_width": bbox["width"],
        "clipped_width": bbox["width"],
    }


def build_respuestas_by_side(answers, visible_pages):
    """
    Separa respuestas por lado (a=izquierda, b=derecha).
    La numeración es CONTINUA: A empieza en 1, B continúa donde A termina.
    Así los números coinciden con los paréntesis (1)...(11), (12)...(23)
    que el OCR devuelve.
    """
    resp_a, resp_b = {}, {}

    if len(visible_pages) < 2:
        for i, item in enumerate(answers):
            resp_a[i+1] = item["text"]
        return {"a": resp_a, "b": {}}

    bbox_a = visible_pages[0]["bbox"]
    bbox_b = visible_pages[1]["bbox"]

    def inside(item, bbox):
        return (
            bbox["x"] <= item["x"] <= bbox["x"] + bbox["width"] and
            bbox["y"] <= item["y"] <= bbox["y"] + bbox["height"]
        )

    a_items = [item for item in answers if inside(item, bbox_a)]
    b_items = [item for item in answers if inside(item, bbox_b)]

    a_items.sort(key=lambda x: (x["y"], x["x"]))
    b_items.sort(key=lambda x: (x["y"], x["x"]))

    # A empieza en 1
    for i, item in enumerate(a_items):
        resp_a[i + 1] = item["text"]

    # B continúa donde A terminó
    b_start = len(a_items) + 1
    for i, item in enumerate(b_items):
        resp_b[b_start + i] = item["text"]

    return {"a": resp_a, "b": resp_b}


def cells_in_bbox(answers, bbox):
    return [
        a for a in answers
        if bbox["x"] <= a["x"] <= bbox["x"] + bbox["width"]
        and bbox["y"] <= a["y"] <= bbox["y"] + bbox["height"]
    ]


def process_pair(page, save_dir, level, page_num, do_ocr):
    print(f"\n  === PAR ({level}{page_num}) ===")

    _, visible = get_page_fingerprint(page)
    if not visible:
        print("     ! Sin paginas visibles")
        return None

    print("     [1] Extrayendo respuestas correctas...")
    toggle_answer_display(page, activate=True)
    page.wait_for_timeout(500)

    answers = extract_answers_from_dom(page)
    resp_by_side = build_respuestas_by_side(answers, visible)

    print(f"         Pagina A: {len(resp_by_side['a'])} respuestas")
    print(f"         Pagina B: {len(resp_by_side['b'])} respuestas")

    print("     [2] Tomando screenshots (con recorte de header)...")
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

        # Celdas de ESTA página para recortar contextualmente
        side_cells = cells_in_bbox(answers, v["bbox"])
        ss_info = take_screenshot_single_page(page, v["bbox"], filepath,
                                              answer_cells=side_cells, side=side)
        print(f"         [SS] {filename}  (clip: {ss_info['clipped_width']}px, "
              f"target: {ss_info['target_width']}px)")

        screenshots.append({
            "filepath": filepath,
            "side": side,
            "label": f"{level}{page_num}{side}",
            "respuestas_correctas": resp_by_side[side],
            "target_width": ss_info["target_width"],
        })

    validations = []

    if do_ocr and MATHPIX_APP_ID and MATHPIX_APP_KEY:
        if COMBINED_OCR_MODE and len(screenshots) == 2:
            # ─── MODO COMBINADO: 1 sola llamada a Mathpix para A+B ───
            print("     [3] OCR en modo COMBINADO (1 llamada para A+B)...")
            validations = run_combined_ocr(screenshots, save_dir, level, page_num)
        else:
            # ─── MODO INDIVIDUAL: 1 llamada por página (fallback) ───
            print("     [3] OCR en modo INDIVIDUAL (1 llamada por pagina)...")
            for ss in screenshots:
                print(f"         Procesando {ss['label']}...")

                ocr_result = ocr_image(
                    ss["filepath"],
                    target_width=ss.get("target_width"),
                    compact_gaps=(ss["side"] == "b"),
                )

                if "error" in ocr_result:
                    print(f"         ERROR: {ocr_result['error']}")
                    continue

                confidence = ocr_result.get("confidence", 0)
                ocr_text = ocr_result.get("text", "")

                if not ocr_text and "line_data" in ocr_result:
                    ocr_text = "\n".join([
                        line.get("text", "") for line in ocr_result["line_data"]
                    ])

                print(f"         Confianza: {confidence:.1%}")
                print(f"\n--- OCR TEXTO ({ss['label']}) ---")
                print(ocr_text)
                print("--------------------------------\n")

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

    return {
        "page_num": page_num,
        "screenshots": screenshots,
        "respuestas": resp_by_side,
        "validations": validations
    }


def run_combined_ocr(screenshots, save_dir, level, page_num):
    """
    Procesa ambas páginas (A y B) en una sola llamada a Mathpix:
    1. Preprocesa cada página en memoria (con su side-specific logic).
    2. Las concatena horizontalmente con separador blanco.
    3. Guarda la imagen combinada en disco.
    4. Una sola llamada a Mathpix.
    5. Parsea el OCR y separa A vs B por los números esperados (1-11 vs 12-23).

    Devuelve lista de validations (igual que el modo individual).
    """
    # Ordenar: A primero (lado izquierdo), B después
    ss_map = {ss["side"]: ss for ss in screenshots}
    ss_a = ss_map.get("a")
    ss_b = ss_map.get("b")

    if not ss_a or not ss_b:
        print("         ! No hay ambas páginas, imposible combinar")
        return []

    # Preprocesar cada página en memoria
    try:
        img_a_rgb = Image.open(ss_a["filepath"]).convert("RGB")
        img_b_rgb = Image.open(ss_b["filepath"]).convert("RGB")

        pp_a = preprocess_pil(img_a_rgb, target_width=ss_a.get("target_width"),
                              compact_gaps=False)
        pp_b = preprocess_pil(img_b_rgb, target_width=ss_b.get("target_width"),
                              compact_gaps=False)

        combined = combine_pages_horizontal(pp_a, pp_b,
                                            separator_px=COMBINED_SEPARATOR_PX)

        combined_path = os.path.join(save_dir, f"{level}{page_num}_combined.png")
        combined.save(combined_path, "PNG", optimize=True)
        print(f"         [COMBINED] {os.path.basename(combined_path)}  "
              f"({combined.width}x{combined.height}px)")
    except Exception as e:
        print(f"         ! Error al combinar paginas: {e}")
        return []

    # OCR directo (sin doble preproceso)
    ocr_result = ocr_file_direct(combined_path)

    if "error" in ocr_result:
        print(f"         ERROR: {ocr_result['error']}")
        return []

    confidence = ocr_result.get("confidence", 0)
    ocr_text = ocr_result.get("text", "")

    if not ocr_text and "line_data" in ocr_result:
        ocr_text = "\n".join([
            line.get("text", "") for line in ocr_result["line_data"]
        ])

    print(f"         Confianza: {confidence:.1%}")
    print(f"\n--- OCR TEXTO COMBINADO ---")
    print(ocr_text)
    print("---------------------------\n")

    # ─── GUARDAR OCR RAW EN ARCHIVO ───
    # Así si hay que corregir algo, solo re-parseas sin llamar a Mathpix
    ocr_cache_path = os.path.join(save_dir, f"{level}{page_num}_ocr.json")
    ocr_cache = {
        "level": level,
        "page_num": page_num,
        "confidence": confidence,
        "ocr_text": ocr_text,
        "ocr_full_response": ocr_result,
    }
    try:
        with open(ocr_cache_path, "w", encoding="utf-8") as f:
            json.dump(ocr_cache, f, indent=2, ensure_ascii=False, default=str)
        print(f"         [CACHE] {os.path.basename(ocr_cache_path)}")
    except Exception as e:
        print(f"         ! No se pudo guardar cache OCR: {e}")

    # Parsear con TODOS los números esperados (A+B juntos)
    all_expected = list(ss_a["respuestas_correctas"].keys()) + \
                   list(ss_b["respuestas_correctas"].keys())
    ejercicios = parse_kumon_exercises(ocr_text, expected_numbers=all_expected)

    # Separar resultados por página según rango de números
    nums_a = set(ss_a["respuestas_correctas"].keys())
    nums_b = set(ss_b["respuestas_correctas"].keys())

    ej_a = {n: v for n, v in ejercicios.items() if n in nums_a}
    ej_b = {n: v for n, v in ejercicios.items() if n in nums_b}

    validations = []

    for (label, ej, resp_correctas) in [
        (ss_a["label"], ej_a, ss_a["respuestas_correctas"]),
        (ss_b["label"], ej_b, ss_b["respuestas_correctas"]),
    ]:
        if ej:
            resultados = comparar_respuestas(ej, resp_correctas)
            correctos, total, fuzzy = mostrar_validacion(resultados,
                                                         f"Validacion {label}")
            validations.append({
                "label": label,
                "correctos": correctos,
                "total": total,
                "fuzzy": fuzzy,
                "resultados": resultados,
                "ocr_raw": ocr_text  # mismo OCR para ambos (es combinado)
            })
        else:
            print(f"         ! No se detectaron ejercicios de {label}")

    return validations


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


def get_all_markboxes_in_viewport(page, visible_pages):
    """
    Obtiene TODAS las .mark-box que caen dentro de las páginas visibles,
    separadas por lado (A/B) y ordenadas por posición Y.
    
    Returns:
        {"a": [lista de locators ordenados por Y],
         "b": [lista de locators ordenados por Y]}
    """
    boxes = page.locator(".mark-box")
    count = boxes.count()
    
    if len(visible_pages) < 2:
        return {"a": [], "b": []}
    
    bbox_a = visible_pages[0]["bbox"]
    bbox_b = visible_pages[1]["bbox"]
    
    a_boxes = []
    b_boxes = []
    
    for i in range(count):
        box = boxes.nth(i)
        try:
            bb = box.bounding_box()
            if not bb or bb["width"] < 5:
                continue
            
            cx = bb["x"] + bb["width"] / 2
            cy = bb["y"] + bb["height"] / 2
            
            # ¿Está dentro de página A?
            if (bbox_a["x"] <= cx <= bbox_a["x"] + bbox_a["width"] and
                bbox_a["y"] <= cy <= bbox_a["y"] + bbox_a["height"]):
                a_boxes.append({"locator": box, "y": bb["y"], "x": bb["x"]})
            # ¿Está dentro de página B?
            elif (bbox_b["x"] <= cx <= bbox_b["x"] + bbox_b["width"] and
                  bbox_b["y"] <= cy <= bbox_b["y"] + bbox_b["height"]):
                b_boxes.append({"locator": box, "y": bb["y"], "x": bb["x"]})
        except:
            pass
    
    a_boxes.sort(key=lambda b: b["y"])
    b_boxes.sort(key=lambda b: b["y"])
    
    return {"a": a_boxes, "b": b_boxes}


def click_to_state(box_locator, target_state, max_clicks=4):
    """
    Clickea una checkbox hasta que su .mark-box-type tenga la clase target_state.
    Estados posibles: 'default', 'triangle', 'check'.
    
    El ciclo de clicks es: default → triangle → check → default → ...
    
    Returns:
        True si logró el estado deseado, False si no.
    """
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
    """
    Marca cada checkbox según el resultado de la validación.
    
    Mapeo por proximidad: cada checkbox se empareja con el .answer-item
    más cercano en Y (dentro del mismo lado A/B). Así sabemos exactamente
    a qué ejercicio corresponde cada checkbox, incluso cuando algunas
    checkboxes desaparecieron porque el ejercicio ya fue calificado
    como correcto en pasadas anteriores.
    
    Requiere que el AnswerDisplay esté ACTIVADO para que los .answer-item
    tengan posiciones válidas.
    """
    # Recopilar resultados por número de ejercicio
    results_by_num = {}
    for v in validations:
        for r in v.get("resultados", []):
            results_by_num[r["numero"]] = r
    
    if not results_by_num:
        print("         ! Sin resultados de validación para marcar")
        return 0
    
    # Activar respuestas para que los .answer-item tengan posiciones
    toggle_answer_display(page, activate=True)
    page.wait_for_timeout(500)
    
    # Extraer posiciones de los answer-items
    answers = extract_answers_from_dom(page)
    
    # Desactivar respuestas (no queremos que se vean mientras marcamos)
    toggle_answer_display(page, activate=False)
    page.wait_for_timeout(300)
    
    # Obtener checkboxes por lado
    cb = get_all_markboxes_in_viewport(page, visible_pages)
    
    if len(visible_pages) < 2:
        print("         ! Menos de 2 páginas visibles")
        return 0
    
    bbox_a = visible_pages[0]["bbox"]
    bbox_b = visible_pages[1]["bbox"]
    
    def inside_bbox(item, bbox):
        return (bbox["x"] <= item["x"] <= bbox["x"] + bbox["width"] and
                bbox["y"] <= item["y"] <= bbox["y"] + bbox["height"])
    
    # Separar answer-items por lado y asociar con su número de ejercicio
    # Los números vienen de resp_by_side que ya tiene la numeración correcta
    a_answers = []
    b_answers = []
    
    # Reconstruir: para cada lado, los items ordenados por Y corresponden
    # a los números ordenados de resp_by_side
    a_items_sorted = [a for a in answers if inside_bbox(a, bbox_a)]
    b_items_sorted = [a for a in answers if inside_bbox(a, bbox_b)]
    a_items_sorted.sort(key=lambda x: x["y"])
    b_items_sorted.sort(key=lambda x: x["y"])
    
    a_nums = sorted(resp_by_side.get("a", {}).keys())
    b_nums = sorted(resp_by_side.get("b", {}).keys())
    
    for i, item in enumerate(a_items_sorted):
        if i < len(a_nums):
            a_answers.append({"y": item["y"], "num": a_nums[i]})
    for i, item in enumerate(b_items_sorted):
        if i < len(b_nums):
            b_answers.append({"y": item["y"], "num": b_nums[i]})
    
    total_marked = 0
    
    for side, side_boxes, side_answers in [("a", cb["a"], a_answers),
                                            ("b", cb["b"], b_answers)]:
        if not side_boxes or not side_answers:
            continue
        
        # Para cada checkbox, encontrar el answer-item más cercano en Y
        for box_info in side_boxes:
            box_y = box_info["y"]
            
            # Buscar answer-item más cercano
            best_match = None
            best_dist = 999999
            for ans in side_answers:
                dist = abs(box_y - ans["y"])
                if dist < best_dist:
                    best_dist = dist
                    best_match = ans
            
            if not best_match or best_dist > 50:
                # Ningún answer-item cerca → skip
                continue
            
            ex_num = best_match["num"]
            result = results_by_num.get(ex_num)
            if not result:
                continue
            
            # Determinar estado objetivo
            if result.get("es_correcto"):
                target = "triangle"
            else:
                target = "check"
            
            # Marcar
            try:
                type_div = box_info["locator"].locator(".mark-box-type")
                current_cls = type_div.get_attribute("class") or ""
                
                if target in current_cls:
                    total_marked += 1
                    continue
                
                if click_to_state(box_info["locator"], target):
                    total_marked += 1
                    estado_label = "OK" if target == "triangle" else "X"
                    print(f"         Ej ({ex_num:2d}): {estado_label}")
                else:
                    print(f"         ! Ej ({ex_num:2d}): no se pudo marcar")
            except Exception as e:
                print(f"         ! Ej ({ex_num:2d}): error: {e}")
    
    return total_marked


def mark_checkboxes(page, mode="triangle"):
    """Versión legacy: marca TODAS las checkboxes con el mismo modo."""
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

    print("\n  Activando vista doble...")
    activate_double_view(page)

    print(f"\n{'='*60}\n  FASE 1: EXTRACCION + SCREENSHOTS\n{'='*60}")
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
            if total_fuzzy > 0: print(f" + {total_fuzzy} revisar", end="")
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

    print(f"\n{'='*60}\n  FASE 2: MARCAR CHECKBOXES\n{'='*60}")
    print("\n  Como marcar?")
    print("    [a] AUTOMATICO (segun resultado del OCR)")
    print("    [c] Todas CORRECTAS (legacy)")
    print("    [i] Todas INCORRECTAS (legacy)")
    print("    [s] Saltar")

    choice = input("  -> ").strip().lower()

    if choice in ("a", ""):
        # ─── MODO AUTOMÁTICO: marcar según validación ───
        print("\n  Marcando segun resultados del OCR...")
        pair_count = 0
        last_fp = None

        while pair_count < len(results):
            fp, visible = get_page_fingerprint(page)
            if fp == last_fp:
                break
            last_fp = fp

            r = results[pair_count]
            marked = mark_checkboxes_by_result(
                page, visible,
                r.get("validations", []),
                r.get("respuestas", {})
            )
            print(f"    Par {pair_count+1}: {marked} checkboxes marcadas")

            pair_count += 1
            if pair_count < len(results):
                go_next_page(page)
                page.wait_for_timeout(500)

        save = input("\n  Guardar calificacion? (s/n) [s]: ").strip().lower()
        if save in ("", "s", "si", "y"):
            complete_marking(page)
            print("  Calificacion guardada")

    elif choice in ("c", "i"):
        # ─── MODO LEGACY: todas iguales ───
        mode = "triangle" if choice == "c" else "check"
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

    print(f"\n{'='*60}\n  Set {level} completado\n  {os.path.abspath(save_dir)}\n{'='*60}")


def main():
    print("=" * 60)
    print("  KUMON GRADING ASSISTANT v5")
    print("=" * 60)
    print(f"  Preproceso: rojo={'ON' if PREPROCESS_STRIP_RED else 'OFF'}, "
          f"upscale={PREPROCESS_UPSCALE}x, gamma={PREPROCESS_GAMMA}")
    print(f"  OCR: modo={'COMBINADO (1 llamada/par)' if COMBINED_OCR_MODE else 'INDIVIDUAL (2 llamadas/par)'}")
    print(f"  Recorte de header dinamico: ON ({HEADER_MARGIN_PX}px margen)")

    ensure_folder(BASE_DIR)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto("https://class-navi.digital.kumon.com/mx/index.html")

        print("\n1) Logueate en Kumon\n2) Abre Instructor Marking\n3) Navega al primer par")
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