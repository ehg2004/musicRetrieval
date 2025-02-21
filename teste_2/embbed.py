import os
import pandas as pd
import soundfile as sf  # For efficient streaming
import openl3
import numpy as np
import gc
import tracemalloc
import objgraph

# Constants
INPUT_CSV = "audio_files.csv"  # Input CSV with audio file paths
EMBEDDINGS_CSV = "openl3_embeddings_segmented_3500.csv"  # Output CSV for embeddings
SEGMENT_DURATION = 15  # Segment duration in seconds
SAMPLE_RATE = 44100  # Desired sample rate
#https://github.com/jordipons/sklearn-audio-transfer-learning/blob/master/FAQ.md
# Step 1: Load the list of audio paths from the previous CSV
data = pd.read_csv(INPUT_CSV)

# Filter out rows without valid audio paths
valid_data = data.dropna(subset=["path"])

# Step 2: Initialize the output CSV with headers if it doesn't exist
if not os.path.exists(EMBEDDINGS_CSV):
    with open(EMBEDDINGS_CSV, mode='w') as f:
        header = ["Audio Name", "Artist Name", "Song Name", "Segment Index"] + [f"Embedding_{i}" for i in range(512)]
        f.write(','.join(header) + '\n')
input_repr, content_type, embedding_size = 'mel128', 'music', 512
model = openl3.models.load_audio_embedding_model(
    input_repr, content_type, embedding_size)


# Step 3: Extract OpenL3 embeddings for each 15-second segment of .wav audio files
i = 0
for index, row in valid_data.iterrows():
    audio_path = row["path"]
    audio_name = os.path.basename(audio_path)
    print(f"Processing: {audio_name}")

    if i == 500 :  # Optional limit for testing
        break

    try:
        # Parse artist and song name
        if '-' in audio_name:
            artist_name, song_name = audio_name.rsplit('-', 1)
            song_name = song_name.rsplit('.', 1)[0]  # Remove file extension
        else:
            artist_name = "Unknown"
            song_name = audio_name.rsplit('.', 1)[0]

        # Open the audio file for streaming
        with sf.SoundFile(audio_path) as audio_file:
            sr = audio_file.samplerate
            segment_samples = SEGMENT_DURATION * sr
            samples = []
            segment_index = 0
            blocks=audio_file.blocks(blocksize=segment_samples, dtype='float32')
            # Stream the audio in chunks
            for block in blocks:
                # Ensure mono audio
                if len(block.shape) > 1:
                    block = np.mean(block, axis=1)

                # Skip if the segment is too short
                if len(block) < segment_samples // 2:
                    continue

                # Extract OpenL3 embeddings for the segment
                embedding, ts = openl3.get_audio_embedding(
                    block,
                    sr,
                    model=model,
                    # input_repr="mel128",
                    # content_type="music",
                    # embedding_size=512,
                    batch_size=32
                )

                # Compute the mean embedding for the segment
                mean_embedding = embedding.mean(axis=0)

                # Create a row of data
                row_data = [audio_name, artist_name, song_name, segment_index] + mean_embedding.tolist()

                # Write the row to the CSV
                with open(EMBEDDINGS_CSV, mode='a') as f:
                    f.write(','.join(map(str, row_data)) + '\n')

                segment_index += 1
                # objgraph.show_most_common_types()
                del mean_embedding
                del embedding
                del row_data
                del ts
            del segment_index
            del samples
            del blocks
            del audio_file
            gc.collect()

        # Clear memory after processing each file
        gc.collect()
    except Exception as e:
        print(f"Error processing {audio_name}: {e}")
    i += 1

print(f"Embedding extraction completed. Results saved to '{EMBEDDINGS_CSV}'.")
