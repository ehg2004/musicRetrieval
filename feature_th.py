import os
import pandas as pd
import librosa
import tsfel
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Constants
INPUT_CSV = "youtube_results.csv"  # CSV produced before
FEATURES_CSV = "audio_features.csv"  # Output CSV for features
NUM_THREADS = 8  # Number of threads to use

# Step 1: Load the list of audio paths from the previous CSV
data = pd.read_csv(INPUT_CSV)

# Filter out rows without valid audio paths
valid_data = data.dropna(subset=["Audio Path"])
total_files = len(valid_data)

# Shared progress counter and lock
progress = 0
progress_lock = Lock()

# Function to process a single audio file
def process_audio(row):
    global progress
    audio_path = row["Audio Path"]
    audio_name = os.path.basename(audio_path)

    try:
        # Load the audio file
        signal, sr = librosa.load(audio_path, sr=None)

        # Extract TSFEL features
        cfg = tsfel.get_features_by_domain()  # Default feature configuration
        extracted_features = tsfel.time_series_features_extractor(cfg, signal, fs=sr, verbose=1)

        # Add audio name to features
        extracted_features.insert(0, "Audio Name", audio_name)

        # Update progress
        with progress_lock:
            progress += 1
            print(f"Processed: {audio_name} ({progress}/{total_files})")

        return extracted_features

    except Exception as e:
        print(f"Error processing {audio_name}: {e}")
        # Update progress on failure as well
        with progress_lock:
            progress += 1
            print(f"Skipped: {audio_name} ({progress}/{total_files})")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Step 2: Process audio files using multiple threads
all_features = []

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    # Submit all rows to the executor
    futures = {executor.submit(process_audio, row): row for _, row in valid_data.iterrows()}

    # Collect results as they complete
    for future in as_completed(futures):
        features = future.result()
        if not features.empty:
            all_features.append(features)

# Combine all feature DataFrames
if all_features:
    features_df = pd.concat(all_features, ignore_index=True)
    # Step 3: Save extracted features to a new CSV
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"Feature extraction completed. Results saved to '{FEATURES_CSV}'.")
else:
    print("No features were extracted.")
