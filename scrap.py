import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the website containing the table
YEAR='2024'
url='https://kworb.net/spotify/songs_'+YEAR+'.html#google_vignette'


# Fetch the webpage content
response = requests.get(url)
if response.status_code == 200:
    html_content = response.text
else:
    print("Failed to retrieve the webpage.")
    exit()

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the table with the specified class
table = soup.find('table', class_='addpos')
if not table:
    print("Table not found on the page.")
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

# Save to CSV
df.to_csv("table_data.csv", index=False)
print("Table data saved to 'table_data.csv'.")
