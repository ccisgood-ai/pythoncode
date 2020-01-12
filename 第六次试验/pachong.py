from selenium import webdriver

driver = webdriver.Chrome()
driver.maximize_window()

driver.implicitly_wait(3)  # 等待3秒

driver.get("https://baidu.com")

driver.quit()