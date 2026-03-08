from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # 1) Abre Class Navi (pon la URL que uses para llegar)
        page.goto("https://class-navi.digital.kumon.com/mx/index.html")  # cambia si aplica
        print("1) Loguéate y navega manualmente a Instructor Marking.")
        input("2) Cuando ya estés en la pantalla de marking, presiona ENTER...")

        # 2) Contar mark-boxes
        mark_boxes = page.locator(".mark-box")
        count = mark_boxes.count()
        print(f"Encontrados {count} mark-boxes")

        # 3) Intentar ponerlos en CHECK (ciclando)
        for i in range(count):
            box = mark_boxes.nth(i)
            # lee estado actual
            type_div = box.locator(".mark-box-type")
            for attempt in range(6):
                cls = type_div.get_attribute("class") or ""
                if "check" in cls:
                    break
                box.click()
                time.sleep(0.15)

            cls_final = type_div.get_attribute("class") or ""
            print(f"[{i}] -> {cls_final}")

        input("Listo. ENTER para cerrar...")
        browser.close()

if __name__ == "__main__":

    run()
