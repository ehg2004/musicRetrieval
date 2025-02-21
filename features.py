import os
import pandas as pd
import librosa
import tsfel

# Constants
INPUT_CSV = "youtube_results.csv"  # CSV produced before
FEATURES_CSV = "audio_features.csv"  # Output CSV for features

# Step 1: Load the list of audio paths from the previous CSV
data = pd.read_csv(INPUT_CSV)

# Filter out rows without valid audio paths
valid_data = data.dropna(subset=["Audio Path"])

# Initialize the DataFrame for features
features_df = pd.DataFrame()

# Step 2: Extract features for each audio file
for index, row in valid_data.iterrows():
    audio_path = row["Audio Path"]
    audio_name = os.path.basename(audio_path)
    print(f"Processing: {audio_name}")

    try:
        # Load the audio file
        signal, sr = librosa.load(audio_path, sr=None)

        # Extract TSFEL features
        cfg = tsfel.get_features_by_domain()  # Default feature configuration
        extracted_features = tsfel.time_series_features_extractor(cfg, signal, fs=sr, verbose=1)

        # Add audio name to features
        extracted_features.insert(0, "Audio Name", audio_name)

        # Append features to the main DataFrame
        features_df = pd.concat([features_df, extracted_features], ignore_index=True)

    except Exception as e:
        print(f"Error processing {audio_name}: {e}")

# Step 3: Save extracted features to a new CSV
features_df.to_csv(FEATURES_CSV, index=False)
print(f"Feature extraction completed. Results saved to '{FEATURES_CSV}'.")
