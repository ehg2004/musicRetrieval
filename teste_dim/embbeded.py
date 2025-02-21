import os
import pandas as pd
import librosa
import torch
from hear21passt.base import get_basic_model

# Constants
INPUT_CSV = "audio_files.csv"  # CSV produced before
FEATURES_CSV = "audio_features_segmented_passt.csv"  # Output CSV for segmented features
SEGMENT_DURATION = 15  # Segment duration in seconds
SAMPLE_RATE = 32000  # Required sampling rate for PaSST

# Load the PaSST model
model = get_basic_model(mode="logits")  # Use "logits" or "features" based on your use case
model.eval()
model = model.cuda()

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
        # Ensure the file format is .wa

        # Load the raw .wav audio file
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, )  # Resample to PaSST's required sampling rate
        total_samples = len(signal)
        segment_samples = SEGMENT_DURATION * SAMPLE_RATE  # Number of samples per 15-second segment

        # Process each segment
        for start_sample in range(0, total_samples, segment_samples):
            end_sample = min(start_sample + segment_samples, total_samples)

            # Extract the segment
            segment_signal = signal[start_sample:end_sample]

            # Skip if the segment is too short
            if len(segment_signal) < segment_samples // 2:
                continue

            # Pad the segment to ensure consistent size
            if len(segment_signal) < segment_samples:
                segment_signal = librosa.util.fix_length(segment_signal, size=segment_samples)

            # Prepare the segment for PaSST
            segment_tensor = torch.tensor(segment_signal, dtype=torch.float32).unsqueeze(0).cuda()  # Shape: [1, samples]
            with torch.no_grad():
                embeddings = model(segment_tensor)  # Extract features (logits or embeddings)

            # Flatten embeddings and store them
            flattened_features = embeddings.cpu().numpy().flatten()
            features_row = {"Audio Name": audio_name, "Segment Index": start_sample // segment_samples}
            for i, feature in enumerate(flattened_features):
                features_row[f"Feature_{i}"] = feature

            # Append to DataFrame
            features_df = pd.concat([features_df, pd.DataFrame([features_row])], ignore_index=True)

    except Exception as e:
        print(f"Error processing {audio_name}: {e}")

# Step 3: Save extracted features to a new CSV
features_df.to_csv(FEATURES_CSV, index=False)
print(f"Feature extraction completed. Results saved to '{FEATURES_CSV}'.")
