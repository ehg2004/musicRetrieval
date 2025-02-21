import os
import pandas as pd
import librosa
import tsfel

# Constants
INPUT_CSV = "youtube_results.csv"  # CSV produced before
FEATURES_CSV = "audio_features_segmented.csv"  # Output CSV for segmented features
SEGMENT_DURATION = 15  # Segment duration in seconds

# Step 1: Load the list of audio paths from the previous CSV
data = pd.read_csv(INPUT_CSV)

# Filter out rows without valid audio paths
valid_data = data.dropna(subset=["Audio Path"])

# Initialize the DataFrame for features
features_df = pd.DataFrame()

# Step 2: Extract features for each 15-second segment of .wav audio files
for index, row in valid_data.iterrows():
    audio_path = row["Audio Path"]
    audio_name = os.path.basename(audio_path)
    print(f"Processing: {audio_name}")

    try:
        # Ensure the file format is .wav
        # if not audio_path.lower().endswith('.wav'):
        #     raise ValueError("Audio file is not in .wav format.")

        # Load the raw .wav audio file
        signal, sr = librosa.load(audio_path, sr=None, )  # Load in original sampling rate

        # Calculate the number of samples per segment
        segment_samples = SEGMENT_DURATION * sr
        total_samples = len(signal)

        # Process each segment
        for start_sample in range(0, total_samples, segment_samples):
            end_sample = min(start_sample + segment_samples, total_samples)

            # Extract the segment
            segment_signal = signal[start_sample:end_sample]

            # Skip if the segment is too short
            if len(segment_signal) < segment_samples // 2:
                continue

            # Extract TSFEL features for the segment
            cfg = tsfel.get_features_by_domain()  # Default feature configuration
            extracted_features = tsfel.time_series_features_extractor(
                cfg, segment_signal, fs=sr, verbose=1
            )

            # Add audio name and segment index to features
            segment_index = start_sample // segment_samples
            extracted_features.insert(0, "Segment Index", segment_index)
            extracted_features.insert(0, "Audio Name", audio_name)

            # Append features to the main DataFrame
            features_df = pd.concat([features_df, extracted_features], ignore_index=True)

    except Exception as e:
        print(f"Error processing {audio_name}: {e}")

# Step 3: Save extracted features to a new CSV
features_df.to_csv(FEATURES_CSV, index=False)
print(f"Feature extraction completed. Results saved to '{FEATURES_CSV}'.")
