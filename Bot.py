from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # 1️⃣ Abrir Class Navi
        page.goto("https://class-navi.digital.kumon.com/mx/index.html")
        print("1) Loguéate y navega manualmente a Instructor Marking.")
        input("2) Cuando estés en la pantalla de marking, presiona ENTER...")

        print("\n▶ Empezando a buscar checkboxes visibles...")

        # 2️⃣ Locator dinámico para checkboxes
        mark_boxes = page.locator(".mark-box")
        count = mark_boxes.count()
        print(f"✅ Encontrados {count} checkboxes visibles\n")

        # 3️⃣ Recorrer cada checkbox
        for i in range(count):
            box = mark_boxes.nth(i)
            # Esperar a que sea visible
            box.wait_for(state="visible")

            type_div = box.locator(".mark-box-type")
            cls = type_div.get_attribute("class") or ""

            if "check" in cls:
                estado = "ya marcado ✅"
            else:
                # Intentar marcar hasta 6 veces
                for attempt in range(6):
                    cls = type_div.get_attribute("class") or ""
                    if "check" in cls:
                        break
                    try:
                        box.click()
                    except:
                        time.sleep(0.1)  # esperar si falla
                    time.sleep(0.15)
                cls = type_div.get_attribute("class") or ""
                estado = "marcado automáticamente ✅" if "check" in cls else "no marcado ❌"

            # Mensaje en la terminal
            print(f"[{i+1}/{count}] -> Estado: {estado}")

        input("\nListo. Presiona ENTER para cerrar...")
        browser.close()

if __name__ == "__main__":
    run()