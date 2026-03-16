from playwright.sync_api import sync_playwright
import os
import time

SCREENSHOT_DIR = "screenshots"


def ensure_folder():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)


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
        is_disabled = down_btn.is_disabled()
        if is_disabled:
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


def take_smart_screenshot(page, screenshot_counter):
    """
    - Ambas con checkboxes → par completo
    - Solo una → solo esa hoja
    - Ninguna → skip
    """
    visible = get_visible_worksheet_pages(page)

    if not visible:
        print("⚠ No se detectaron hojas visibles.")
        return screenshot_counter, visible

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
        return screenshot_counter, visible

    if len(pages_with_cb) == 2:
        # Ambas tienen → par completo
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
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        }

        screenshot_counter += 1
        path = os.path.join(
            SCREENSHOT_DIR,
            f"pair_{screenshot_counter:02d}.png"
        )
        page.screenshot(path=path, clip=clip)
        print(f"📸 Par completo: {path}")

    elif len(pages_with_cb) == 1:
        # Solo una → individual
        bb = pages_with_cb[0]["bbox"]

        clip = {
            "x": bb["x"],
            "y": bb["y"],
            "width": bb["width"],
            "height": bb["height"]
        }

        screenshot_counter += 1
        path = os.path.join(
            SCREENSHOT_DIR,
            f"sheet_{screenshot_counter:02d}.png"
        )
        page.screenshot(path=path, clip=clip)
        print(f"📸 Hoja individual: {path}")

    return screenshot_counter, visible


def run():
    ensure_folder()

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
            "2) Déjalo en la primera hoja y "
            "presiona ENTER..."
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

        print("\nActivando vista doble...")
        activate_double_view(page)

        print(
            "\n🔄 Recorriendo hojas automáticamente...\n"
        )

        screenshot_counter = 0
        pair_number = 0
        last_fingerprint = None

        while True:
            pair_number += 1
            print(
                f"{'='*10} PAR {pair_number} {'='*10}"
            )

            page.wait_for_timeout(500)

            # Fingerprint para detectar si cambió
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

            # Tomar screenshots según checkboxes
            screenshot_counter, _ = (
                take_smart_screenshot(
                    page, screenshot_counter
                )
            )

            # Avanzar
            print(f"\n⏩ Avanzando...")
            moved = go_next_page(page)

            if not moved:
                print("🏁 No se pudo avanzar → FIN")
                break

            page.wait_for_timeout(1000)

            # Verificar cambio real
            new_fp, _ = get_page_fingerprint(page)

            if new_fp and new_fp == last_fingerprint:
                page.wait_for_timeout(1500)
                new_fp, _ = get_page_fingerprint(page)

                if new_fp == last_fingerprint:
                    print("🏁 Sin más páginas → FIN")
                    break

            print()

        print(f"\n{'='*40}")
        print(f"✅ Proceso terminado")
        print(f"📄 Pares recorridos: {pair_number}")
        print(
            f"📸 Screenshots tomados: "
            f"{screenshot_counter}"
        )
        print(
            f"📁 Carpeta: "
            f"{os.path.abspath(SCREENSHOT_DIR)}"
        )

        input("\nPresiona ENTER para cerrar...")
        browser.close()


if __name__ == "__main__":
    run()