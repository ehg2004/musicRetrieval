import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from yt_dlp import YoutubeDL
import csv
import os
import time
import pandas as pd

# Constants
YEAR='2024'
TABLE_URL='https://kworb.net/spotify/songs_'+YEAR+'.html#google_vignette'
YOUTUBE_URL = "https://www.youtube.com"
OUTPUT_DIR = "downloads"  # Directory to save audio files
CSV_FILE = "youtube_results.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Scrape the table
response = requests.get(TABLE_URL)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table and extract the first column
table = soup.find('table', class_='addpos')
if not table:
    print("Table not found!")
    exit()


# Extract table headers
headers = [header.text.strip() for header in table.find_all('th')]

# Extract table rows
rows = []
for row in table.find_all('tr'):
    cells = row.find_all(['td', 'th'])
    row_data = [cell.text.strip() for cell in cells]
    if row_data:  # Avoid empty rows
        rows.append(row_data)

# Create a DataFrame for better representation
df = pd.DataFrame(rows, columns=headers)

# Extract the first column (titles)
titles = []
for row in table.find_all('tr')[1:]:  # Skip header
    cells = row.find_all('td')
    if cells:
        titles.append(cells[0].text.strip())

# Check if titles were found
if not titles:
    print("No titles found!")
    exit()

# Initialize the CSV file with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Title", "URL", "Audio Path"])
        writer.writeheader()

# Step 2: Search each title on YouTube and download audio
driver = webdriver.Chrome()  # Ensure the correct WebDriver is installed
driver.get(YOUTUBE_URL)

for title in titles:
    # Search for the title
    search_box = driver.find_element(By.NAME, "search_query")
    search_box.clear()
    search_box.send_keys(title)
    search_box.send_keys(Keys.RETURN)
    time.sleep(2)  # Wait for results to load

    # Click on the first result
    try:
        first_video = driver.find_element(By.XPATH, '(//ytd-video-renderer//a[@id="video-title"])[1]')
        video_url = first_video.get_attribute("href")
        print(f"Processing video for: {title}")
        print(f"URL: {video_url}")

        # Download audio using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(OUTPUT_DIR, f"{title}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            downloaded_file = os.path.join(OUTPUT_DIR, f"{title}.mp3")

        # Append results to the CSV
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Title", "URL", "Audio Path"])
            writer.writerow({"Title": title, "URL": video_url, "Audio Path": downloaded_file})

    except Exception as e:
        print(f"Could not process video for: {title}. Error: {e}")
        # Log failed attempts in the CSV
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Title", "URL", "Audio Path"])
            writer.writerow({"Title": title, "URL": None, "Audio Path": None})

driver.quit()

print(f"Processing completed. Results saved in '{CSV_FILE}'.")
