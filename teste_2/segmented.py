import os
import pandas as pd
import soundfile as sf
import tsfel
import gc

# Constants
INPUT_CSV = "audio_files.csv"  # Input CSV with audio file paths
FEATURES_CSV = "audio_features_segmented_500.csv"  # Output CSV for segmented features
SEGMENT_DURATION = 15  # Segment duration in seconds

# Step 1: Load the list of audio paths from the previous CSV
data = pd.read_csv(INPUT_CSV)
cfg = tsfel.load_json('features1.json')  #

# Filter out rows without valid audio paths
valid_data = data.dropna(subset=["path"])

# Step 2: Initialize the output CSV with headers if it doesn't exist
if not os.path.exists(FEATURES_CSV):
    with open(FEATURES_CSV, mode='w') as f:
        header = ["Audio Name", "Artist Name", "Song Name", "Segment Index", "0_Absolute energy,0_Area under the curve,0_Autocorrelation,0_Average power,0_Centroid,0_ECDF Percentile Count_0,0_ECDF Percentile Count_1,0_ECDF Percentile_0,0_ECDF Percentile_1,0_ECDF_0,0_ECDF_1,0_ECDF_2,0_ECDF_3,0_ECDF_4,0_ECDF_5,0_ECDF_6,0_ECDF_7,0_ECDF_8,0_ECDF_9,0_Entropy,0_Fundamental frequency,0_Histogram mode,0_Human range energy,0_Interquartile range,0_Kurtosis,0_LPCC_0,0_LPCC_1,0_LPCC_10,0_LPCC_11,0_LPCC_2,0_LPCC_3,0_LPCC_4,0_LPCC_5,0_LPCC_6,0_LPCC_7,0_LPCC_8,0_LPCC_9,0_MFCC_0,0_MFCC_1,0_MFCC_10,0_MFCC_11,0_MFCC_2,0_MFCC_3,0_MFCC_4,0_MFCC_5,0_MFCC_6,0_MFCC_7,0_MFCC_8,0_MFCC_9,0_Max,0_Max power spectrum,0_Maximum frequency,0_Mean,0_Mean absolute deviation,0_Mean absolute diff,0_Mean diff,0_Median,0_Median absolute deviation,0_Median absolute diff,0_Median diff,0_Median frequency,0_Min,0_Negative turning points,0_Neighbourhood peaks,0_Peak to peak distance,0_Positive turning points,0_Power bandwidth,0_Root mean square,0_Signal distance,0_Skewness,0_Slope,0_Spectral centroid,0_Spectral decrease,0_Spectral distance,0_Spectral entropy,0_Spectral kurtosis,0_Spectral positive turning points,0_Spectral roll-off,0_Spectral roll-on,0_Spectral skewness,0_Spectral slope,0_Spectral spread,0_Spectral variation,0_Spectrogram mean coefficient_0.0Hz,0_Spectrogram mean coefficient_10064.52Hz,0_Spectrogram mean coefficient_10838.71Hz,0_Spectrogram mean coefficient_11612.9Hz,0_Spectrogram mean coefficient_12387.1Hz,0_Spectrogram mean coefficient_13161.29Hz,0_Spectrogram mean coefficient_13935.48Hz,0_Spectrogram mean coefficient_14709.68Hz,0_Spectrogram mean coefficient_1548.39Hz,0_Spectrogram mean coefficient_15483.87Hz,0_Spectrogram mean coefficient_16258.06Hz,0_Spectrogram mean coefficient_17032.26Hz,0_Spectrogram mean coefficient_17806.45Hz,0_Spectrogram mean coefficient_18580.65Hz,0_Spectrogram mean coefficient_19354.84Hz,0_Spectrogram mean coefficient_20129.03Hz,0_Spectrogram mean coefficient_20903.23Hz,0_Spectrogram mean coefficient_21677.42Hz,0_Spectrogram mean coefficient_22451.61Hz,0_Spectrogram mean coefficient_2322.58Hz,0_Spectrogram mean coefficient_23225.81Hz,0_Spectrogram mean coefficient_24000.0Hz,0_Spectrogram mean coefficient_3096.77Hz,0_Spectrogram mean coefficient_3870.97Hz,0_Spectrogram mean coefficient_4645.16Hz,0_Spectrogram mean coefficient_5419.35Hz,0_Spectrogram mean coefficient_6193.55Hz,0_Spectrogram mean coefficient_6967.74Hz,0_Spectrogram mean coefficient_774.19Hz,0_Spectrogram mean coefficient_7741.94Hz,0_Spectrogram mean coefficient_8516.13Hz,0_Spectrogram mean coefficient_9290.32Hz,0_Standard deviation,0_Sum absolute diff,0_Variance,0_Wavelet absolute mean_12000.0Hz,0_Wavelet absolute mean_1333.33Hz,0_Wavelet absolute mean_1500.0Hz,0_Wavelet absolute mean_1714.29Hz,0_Wavelet absolute mean_2000.0Hz,0_Wavelet absolute mean_2400.0Hz,0_Wavelet absolute mean_3000.0Hz,0_Wavelet absolute mean_4000.0Hz,0_Wavelet absolute mean_6000.0Hz,0_Wavelet energy_12000.0Hz,0_Wavelet energy_1333.33Hz,0_Wavelet energy_1500.0Hz,0_Wavelet energy_1714.29Hz,0_Wavelet energy_2000.0Hz,0_Wavelet energy_2400.0Hz,0_Wavelet energy_3000.0Hz,0_Wavelet energy_4000.0Hz,0_Wavelet energy_6000.0Hz,0_Wavelet entropy,0_Wavelet standard deviation_12000.0Hz,0_Wavelet standard deviation_1333.33Hz,0_Wavelet standard deviation_1500.0Hz,0_Wavelet standard deviation_1714.29Hz,0_Wavelet standard deviation_2000.0Hz,0_Wavelet standard deviation_2400.0Hz,0_Wavelet standard deviation_3000.0Hz,0_Wavelet standard deviation_4000.0Hz,0_Wavelet standard deviation_6000.0Hz,0_Wavelet variance_12000.0Hz,0_Wavelet variance_1333.33Hz,0_Wavelet variance_1500.0Hz,0_Wavelet variance_1714.29Hz,0_Wavelet variance_2000.0Hz,0_Wavelet variance_2400.0Hz,0_Wavelet variance_3000.0Hz,0_Wavelet variance_4000.0Hz,0_Wavelet variance_6000.0Hz,0_Zero crossing rate"]
        f.write(','.join(header) + '\n')

# Default feature configuration

# Step 3: Process each audio file
i = 0
for index, row in valid_data.iterrows():
    audio_path = row["path"]
    audio_name = os.path.basename(audio_path)
    print(f"Processing: {audio_name}")

    if i == 500:  # Optional limit for testing
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
            segment_index = 0

            # Stream the audio in chunks
            for block in audio_file.blocks(blocksize=segment_samples, dtype='float32'):
                # Ensure mono audio
                if len(block.shape) > 1:
                    block = block.mean(axis=1)

                # Skip if the segment is too short
                if len(block) < segment_samples // 2:
                    continue

                # Extract TSFEL features for the segment
                extracted_features = tsfel.time_series_features_extractor(
                    cfg, block, fs=sr, verbose=1
                )

                # Add audio metadata to the features
                extracted_features.insert(0, "Segment Index", segment_index)
                extracted_features.insert(0, "Song Name", song_name)
                extracted_features.insert(0, "Artist Name", artist_name)
                extracted_features.insert(0, "Audio Name", audio_name)

                # Write the features directly to the CSV
                with open(FEATURES_CSV, mode='a') as f:
                    f.write(','.join(map(str, extracted_features.values.flatten())) + '\n')

                # Increment segment index
                segment_index += 1

                # Clear memory after processing each segment
                del extracted_features
                del block
                gc.collect()

    except Exception as e:
        print(f"Error processing {audio_name}: {e}")
    i += 1

print(f"Feature extraction completed. Results saved to '{FEATURES_CSV}'.")
