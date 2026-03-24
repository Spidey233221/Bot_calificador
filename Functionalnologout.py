from playwright.sync_api import sync_playwright
import os
import time
import re

BASE_DIR = "base_data"


def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


def parse_level_page(user_input):
    user_input = user_input.strip().upper()
    match = re.match(r'^([A-Z0-9]+?)(\d+)$', user_input)
    if not match:
        return None, None
    level = match.group(1)
    start_page = int(match.group(2))
    return level, start_page


def get_visible_worksheet_pages(page):
    visible_pages = []
    pages = page.locator(".worksheet-group-page")
    count = pages.count()

    viewport = page.viewport_size
    if not viewport:
        viewport = {"width": 1600, "height": 1000}

    vw = viewport["width"]
    vh = viewport["height"]

    for i in range(count):
        try:
            item = pages.nth(i)
            bb = item.bounding_box()

            if not bb:
                continue
            if bb["width"] < 150 or bb["height"] < 200:
                continue

            x1 = bb["x"]
            y1 = bb["y"]
            x2 = bb["x"] + bb["width"]
            y2 = bb["y"] + bb["height"]

            inter_w = max(0, min(x2, vw) - max(x1, 0))
            inter_h = max(0, min(y2, vh) - max(y1, 0))

            visible_area = inter_w * inter_h
            total_area = bb["width"] * bb["height"]

            if total_area <= 0:
                continue

            ratio = visible_area / total_area

            if ratio > 0.6:
                visible_pages.append({
                    "index": i,
                    "bbox": bb,
                    "visible_ratio": ratio,
                    "element": item
                })
        except Exception:
            pass

    visible_pages.sort(
        key=lambda v: (v["bbox"]["y"], v["bbox"]["x"])
    )
    return visible_pages


def smart_click(locator, name):
    try:
        locator.click(timeout=3000)
        return True
    except Exception:
        pass
    try:
        locator.click(force=True, timeout=3000)
        return True
    except Exception:
        pass
    try:
        bb = locator.bounding_box()
        if bb:
            x = bb["x"] + bb["width"] / 2
            y = bb["y"] + bb["height"] / 2
            locator.page.mouse.click(x, y)
            return True
    except Exception:
        pass
    print(f"❌ No se pudo clickear {name}")
    return False


def activate_double_view(page):
    both_btn = page.locator("#BothSidesDisplayButton")
    ok = smart_click(both_btn, "BothSidesDisplayButton")
    page.wait_for_timeout(1500)
    return ok


def go_next_page(page):
    down_btn = page.locator("button.down.pager-button")
    try:
        if down_btn.is_disabled():
            return False
        cls = down_btn.get_attribute("class") or ""
        if "disabled" in cls:
            return False
    except Exception:
        pass
    ok = smart_click(down_btn, "Down pager button")
    page.wait_for_timeout(2000)
    return ok


def get_page_fingerprint(page):
    visible = get_visible_worksheet_pages(page)
    if not visible:
        return None, visible
    fp = tuple(
        (v["index"], round(v["bbox"]["y"], 1))
        for v in visible
    )
    return fp, visible


def check_page_has_checkboxes(page, page_bbox):
    mark_boxes = page.locator(".mark-box")
    try:
        count = mark_boxes.count()
    except Exception:
        return False, 0

    found = 0
    px1 = page_bbox["x"]
    py1 = page_bbox["y"]
    px2 = page_bbox["x"] + page_bbox["width"]
    py2 = page_bbox["y"] + page_bbox["height"]

    for i in range(count):
        try:
            box = mark_boxes.nth(i)
            if not box.is_visible():
                continue
            bb = box.bounding_box()
            if not bb:
                continue
            cx = bb["x"] + bb["width"] / 2
            cy = bb["y"] + bb["height"] / 2
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                found += 1
        except Exception:
            pass

    return found > 0, found


def mark_all_visible_checkboxes(page):
    mark_boxes = page.locator(".mark-box")
    try:
        count = mark_boxes.count()
    except Exception:
        return 0

    marked = 0
    for i in range(count):
        try:
            box = mark_boxes.nth(i)
            if not box.is_visible():
                continue

            type_div = box.locator(".mark-box-type")
            cls = type_div.get_attribute("class") or ""

            if "check" in cls:
                marked += 1
                continue

            for attempt in range(6):
                cls = (
                    type_div.get_attribute("class") or ""
                )
                if "check" in cls:
                    break
                try:
                    box.click()
                except Exception:
                    time.sleep(0.1)
                time.sleep(0.15)

            cls = type_div.get_attribute("class") or ""
            if "check" in cls:
                marked += 1
        except Exception:
            pass

    return marked


def take_smart_screenshot(
    page, screenshot_counter, save_dir,
    level, current_page_num
):
    visible = get_visible_worksheet_pages(page)

    if not visible:
        print("⚠ No se detectaron hojas visibles.")
        return (
            screenshot_counter, visible,
            current_page_num
        )

    target_pages = visible[:2]
    pages_with_cb = []

    for v in target_pages:
        has_cb, cb_count = check_page_has_checkboxes(
            page, v["bbox"]
        )
        if has_cb:
            pages_with_cb.append(v)
            print(
                f"   ☑ Hoja index {v['index']}: "
                f"{cb_count} checkboxes"
            )
        else:
            print(
                f"   ⬜ Hoja index {v['index']}: "
                f"sin checkboxes"
            )

    if not pages_with_cb:
        print("⏭ Sin checkboxes → SKIP")
        return (
            screenshot_counter, visible,
            current_page_num
        )

    if len(pages_with_cb) == 2:
        bboxes = [v["bbox"] for v in pages_with_cb]
        x_min = min(bb["x"] for bb in bboxes)
        y_min = min(bb["y"] for bb in bboxes)
        x_max = max(
            bb["x"] + bb["width"] for bb in bboxes
        )
        y_max = max(
            bb["y"] + bb["height"] for bb in bboxes
        )

        clip = {
            "x": x_min, "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        }

        page_a = f"{level}{current_page_num}a"
        page_b = f"{level}{current_page_num}b"
        filename = f"{page_a}_{page_b}.png"

        screenshot_counter += 1
        path = os.path.join(save_dir, filename)
        page.screenshot(path=path, clip=clip)
        print(f"📸 Par: {path}")
        current_page_num += 1

    elif len(pages_with_cb) == 1:
        bb = pages_with_cb[0]["bbox"]
        clip = {
            "x": bb["x"], "y": bb["y"],
            "width": bb["width"],
            "height": bb["height"]
        }

        all_bboxes = [v["bbox"] for v in target_pages]
        is_left = (
            pages_with_cb[0]["bbox"]["x"]
            == min(b["x"] for b in all_bboxes)
        )
        side = "a" if is_left else "b"

        filename = (
            f"{level}{current_page_num}{side}.png"
        )

        screenshot_counter += 1
        path = os.path.join(save_dir, filename)
        page.screenshot(path=path, clip=clip)
        print(f"📸 Hoja individual: {path}")

        if side == "b" or len(target_pages) == 1:
            current_page_num += 1

    return (
        screenshot_counter, visible,
        current_page_num
    )


def process_one_set(page):
    level_input = input(
        "\n📝 Nivel y página inicial "
        "(ej: A11, B5, 2A131): "
    ).strip()

    level, start_page = parse_level_page(level_input)

    if not level or not start_page:
        print(
            "❌ Formato inválido. Usa algo como "
            "A11, B5, C21"
        )
        return False

    print(
        f"   Nivel: {level} | "
        f"Página inicial: {start_page}"
    )

    save_dir = os.path.join(BASE_DIR, level)
    ensure_folder(save_dir)

    print("\n   Activando vista doble...")
    activate_double_view(page)
    page.wait_for_timeout(1000)

    # ════════════════════════════════════
    #  FASE 1: Screenshots de TODAS
    #  las hojas disponibles
    # ════════════════════════════════════
    print(
        "\n🔄 FASE 1: Tomando screenshots de "
        "todas las hojas...\n"
    )

    screenshot_counter = 0
    pair_number = 0
    current_page_num = start_page
    last_fingerprint = None

    while True:
        pair_number += 1
        print(
            f"{'='*10} PAR {pair_number} "
            f"(Página {level}{current_page_num}) "
            f"{'='*10}"
        )

        page.wait_for_timeout(500)

        current_fp, _ = get_page_fingerprint(page)

        if (
            current_fp
            and current_fp == last_fingerprint
        ):
            print(
                "🏁 Contenido igual al anterior → FIN"
            )
            break

        last_fingerprint = current_fp

        (
            screenshot_counter, _,
            current_page_num
        ) = take_smart_screenshot(
            page, screenshot_counter, save_dir,
            level, current_page_num
        )

        # Avanzar
        print(f"\n⏩ Avanzando...")
        moved = go_next_page(page)

        if not moved:
            print("🏁 No se pudo avanzar → FIN")
            break

        page.wait_for_timeout(1000)

        new_fp, _ = get_page_fingerprint(page)

        if new_fp and new_fp == last_fingerprint:
            page.wait_for_timeout(1500)
            new_fp, _ = get_page_fingerprint(page)
            if new_fp == last_fingerprint:
                print("🏁 Sin más páginas → FIN")
                break

        print()

    # ════════════════════════════════════
    #  FASE 2: Marcar checkboxes
    # ════════════════════════════════════
    print(
        f"\n{'='*40}"
        f"\n🔄 FASE 2: Marcando checkboxes...\n"
    )

    confirm = input(
        "   ¿Marcar todas las checkboxes como "
        "correctas? (s/n): "
    ).strip().lower()

    if confirm in ("s", "si", "y", "yes", ""):
        print("   Regresando al inicio del set...")

        for _ in range(pair_number):
            up_btn = page.locator(
                "button.up.pager-button"
            )
            try:
                smart_click(up_btn, "Up button")
                page.wait_for_timeout(800)
            except Exception:
                break

        page.wait_for_timeout(1000)

        mark_pair = 0
        last_fp_mark = None

        while mark_pair < pair_number:
            mark_pair += 1
            print(
                f"   Marcando par {mark_pair}/"
                f"{pair_number}..."
            )

            page.wait_for_timeout(500)

            fp_now, _ = get_page_fingerprint(page)
            if fp_now and fp_now == last_fp_mark:
                break
            last_fp_mark = fp_now

            marked = mark_all_visible_checkboxes(page)
            print(
                f"   ✅ {marked} checkboxes marcadas"
            )

            if mark_pair < pair_number:
                go_next_page(page)
                page.wait_for_timeout(500)

        print("\n✅ Todas las checkboxes marcadas")
    else:
        print("⏭ Marcado omitido")

    print(f"\n{'='*40}")
    print(
        f"📸 Screenshots tomados: "
        f"{screenshot_counter}"
    )
    print(
        f"📁 Guardados en: "
        f"{os.path.abspath(save_dir)}"
    )

    return True


def run():
    ensure_folder(BASE_DIR)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", headless=False
        )
        page = browser.new_page(
            viewport={"width": 1600, "height": 1000}
        )

        page.goto(
            "https://class-navi.digital.kumon.com"
            "/mx/index.html"
        )

        print("1) Loguéate y entra a Instructor Marking.")
        input(
            "2) Déjalo en la primera hoja del set "
            "y presiona ENTER..."
        )

        try:
            page.add_style_tag(
                content=(
                    "::-webkit-scrollbar "
                    "{ display: none !important; }"
                )
            )
        except Exception:
            pass

        while True:
            process_one_set(page)

            print(f"\n{'='*40}")
            choice = input(
                "\n¿Qué deseas hacer?\n"
                "  [s] Procesar otro set\n"
                "  [q] Salir\n"
                "  → "
            ).strip().lower()

            if choice in ("q", "salir", "exit", "n"):
                break

            print(
                "\n📌 Navega al siguiente set "
                "en el navegador."
            )
            input(
                "   Cuando estés listo, "
                "presiona ENTER..."
            )

        print("\n👋 ¡Hasta luego!")
        browser.close()


if __name__ == "__main__":
    run()