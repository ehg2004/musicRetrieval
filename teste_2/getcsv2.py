import os
import csv

# Path to the folder containing the audio files
folder_path = 'downloads/'

# Open a CSV file for writing
with open('audio_files.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header row
    writer.writerow(['title', 'path'])
    
    # Loop through the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Extract the title (without extension)
            title, ext = os.path.splitext(filename)
            
            # Write the title and path to the CSV file
            writer.writerow([title, file_path])

print("CSV file 'audio_files.csv' has been written successfully.")
