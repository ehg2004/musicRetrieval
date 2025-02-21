import os
import csv
from pydub.utils import mediainfo

# Define the relative folder path where your audio files are stored
folder_path = 'downloads'

# Function to get audio duration
def get_audio_duration(file_path):
    info = mediainfo(file_path)
    return float(info['duration']) if 'duration' in info else 0

# Create/open the CSV file to write the data
csv_filename = 'audio_files_info.csv'
with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['title', 'path', 'duration'])  # Write header
    
    # Iterate through the files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if it's an audio file (you can add more extensions if needed)
            if file.endswith(('.mp3', '.wav', '.flac', '.aac')):
                file_path = os.path.join(root, file)
                title = os.path.splitext(file)[0]  # Get the title without extension
                duration = get_audio_duration(file_path)
                writer.writerow([title, os.path.abspath(file_path), duration])

print(f'CSV file "{csv_filename}" has been created successfully.')
