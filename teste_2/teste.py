import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from yt_dlp import YoutubeDL
import pandas as pd
import os
import time


# URL of the website containing the table
YEAR='2024'
TABLE_URL='https://kworb.net/spotify/songs_'+YEAR+'.html#google_vignette'

# Constants
# TABLE_URL = "https://example.com"  # Replace with the actual URL
YOUTUBE_URL = "https://www.youtube.com"
OUTPUT_DIR = "downloads"  # Directory to save audio files

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
titles = []

for i in range(2010,2024):

    YEAR=str(i)
    TABLE_URL='https://kworb.net/spotify/songs_'+YEAR+'.html#google_vignette'

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
    for row in table.find_all('tr')[1:]:  # Skip header
        cells = row.find_all('td')
        if cells:
            titles.append(cells[0].text.strip())


# Save to CSV
df.to_csv("table_data.csv", index=False)
print("Table data saved to 'table_data.csv'.")


# Check if titles were found
if not titles:
    print("No titles found!")
    exit()

# Initialize data for CSV
results = []

# Step 2: Search each title on YouTube and download audio
driver = webdriver.Firefox()  # Ensure the correct WebDriver is installed
driver.get(YOUTUBE_URL)
i = 0
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
                'preferredcodec': 'mp3',  # Save as raw PCM audio in .wav format
                'preferredquality': '192',
            }],
            # 'postprocessor_args': [
            #     '-ss', '60',  # Start at 60 seconds (1:00)
            #     '-t', '60',   # Extract for 60 seconds (1 minute)
            #         ],
        }



        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            downloaded_file = os.path.join(OUTPUT_DIR, f"{title}.mp3")

        # Add to results
        results.append({'Title': title, 'URL': video_url, 'Audio Path': downloaded_file})

    except Exception as e:
        print(f"Could not process video for: {title}. Error: {e}")
        results.append({'Title': title, 'URL': None, 'Audio Path': None})
    i=i+1
    # if i==50:
    #     break
driver.quit()

# Step 3: Save results to a CSV file
csv_file = "youtube_results.csv"
df = pd.DataFrame(results)
df.to_csv(csv_file, index=False)
print(f"Results saved to '{csv_file}'.")
