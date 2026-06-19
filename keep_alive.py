from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import sys

STREAMLIT_URL = "https://sem-cathode-analyzer.streamlit.app/"


def main():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        print(f"Visiting {STREAMLIT_URL}")
        driver.get(STREAMLIT_URL)

        try:
            wake_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'Yes')]"))
            )
            print("Wake-up button found. Clicking it...")
            wake_button.click()

            WebDriverWait(driver, 30).until(
                EC.invisibility_of_element(wake_button)
            )
            print("App woken up successfully ✅")

        except TimeoutException:
            print("No wake-up button found. App is already awake ✅")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        driver.quit()
        print("Script finished.")


if __name__ == "__main__":
    main()
