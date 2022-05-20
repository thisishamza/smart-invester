import csv
import datetime
import time
from datetime import timedelta

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select


def driver_server():
    display = ''
    options = Options()
    options.add_argument('--disable-notifications')
    options.add_argument("--start-maximized")
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--shm-size=2g')
    options.add_argument('--no-sandbox')
    options.add_argument('User-Agent=[Mozilla/5.0]')
    while True:
        try:
            driver = webdriver.Chrome(
                executable_path="chromedriver.exe",
                chrome_options=options
            )

            break
        except Exception as e:
            print(e)
            continue
    return driver


server_dir_path = ""
stocks_data = []
stock_df = pd.DataFrame(
    columns=['time', 'ldcp', 'open', 'high', 'low', 'close', 'volume', 'tic'])


def pakistan_stock(month, tic):
    global stock_df
    global stocks_data
    sitelink = "https://dps.psx.com.pk/historical"
    page = 1
    driver = driver_server()
    driver.get(sitelink)
    driver.find_element_by_xpath('//*[@id="historicalSymbolSearch"]').clear()
    driver.find_element_by_xpath(
        '//*[@id="historicalSymbolSearch"]').send_keys(tic)
    select = Select(driver.find_element_by_css_selector(
        'div.dropdown.historical__month select.dropdown__select'))
    select.select_by_visible_text(month)
    select2 = Select(driver.find_element_by_css_selector(
        'div.dropdown.historical__year select.dropdown__select'))
    select2.select_by_visible_text('2022')
    time.sleep(1)
    driver.find_element_by_xpath('//*[@id="historicalSymbolBtn"]').click()
    time.sleep(3)
    with open('' + f'{tic}.csv', mode='a') as employee_file:
        employee_writer = csv.writer(
            employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        while True:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            div = soup.find('div', {'class': 'dataTables_paginate'})
            span = div.find('span')
            a = span.find_all('a')
            try:
                total_page = int(a[-1].text)
            except:
                break
            print(total_page)
            table = soup.find('table', {'class': 'tbl'})
            tbody = table.find('tbody')
            for rows in tbody.find_all('tr'):
                col = rows.find_all('td')
                sym = tic
                print(sym)
                timee = col[0].text.strip()
                print(timee)
                OPEN = col[1].text.strip()
                print(OPEN)
                high = col[2].text.strip()
                print(high)
                low = col[3].text.strip()
                print(low)
                close = col[4].text.strip()
                print(close)
                volume = col[5].text.strip()
                print(volume)
                try:
                    employee_writer.writerow(
                        [sym, timee, OPEN, high, low, close, volume])
                    stocks_data.append(
                        (sym, timee, OPEN, high, low, close, volume))

                except:
                    pass
            try:
                page = 100
                if total_page >= page:
                    driver.find_element_by_xpath(
                        '//*[@id="historicalTable_next"]').click()
                    time.sleep(2)
                    page = page + 1
                elif page > total_page:
                    df = pd.DataFrame(
                        stocks_data,
                        columns=['tic', 'time', 'open',
                                 'high', 'low', 'close', 'volume']
                    )
                    stock_df = stock_df.append(df, ignore_index=True)
                    driver.close()
                    driver.quit()
                    break
            except:
                pass


list_dates = []
cur_date_time = datetime.datetime.now() - timedelta(days=1)
cur_date = cur_date_time.strftime('%Y-%m-%d')

for _ in range(0, 365):
    list_dates.append(cur_date)
    cur_date_time = cur_date_time - timedelta(days=1)
    cur_date = cur_date_time.strftime('%Y-%m-%d')


def get_new_data(tic):
    list_dates = ['January', 'February', 'March', 'April', 'May',
                  'June', 'July', 'September', 'October', 'November', 'December']
    for date in list_dates:
        pakistan_stock(date, tic)

    data_df = pd.read_csv(f'{tic}.csv')
    return data_df


tic_list = ['APL']
for tic in tic_list:
    get_new_data(tic)
