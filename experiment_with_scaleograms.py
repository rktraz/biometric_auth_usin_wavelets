import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os


import pywt
import numpy as np

def _create_scaleogram(signal: np.ndarray, n=8, wavelet="mexh") -> np.ndarray:
    """Creates scaleogram for signal, and returns it.

    The resulting scaleogram represents scale in the first dimension, time in
    the second dimension, and the color shows amplitude.
    """

    scale_list = np.arange(start=0, stop=len(signal)) / n + 1
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]
    return scaleogram

user_ids = [847, 284]
for user_id in user_ids:
    user_data = df[df['user_id'] == user_id].head(5000)
    save_dir = f"testing_scaleograms/user_{user_id}"
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Create windows of size 225 with a shift factor of 30
    window_size = 225
    shift_factor = 30 # 10

    scale_list = [4, 8, 16, 32]
    wavelet_list = ['mexh', 'morl'] # breaks on "haar"

    for scale in scale_list:
        for wavelet in wavelet_list:
            # Step 3: Create scaleograms for each axis for every window
            x_axis_scaleograms = []
            y_axis_scaleograms = []
            z_axis_scaleograms = []

            for i in range(0, len(user_data) - window_size + 1, shift_factor):
                window = user_data.iloc[i:i+window_size]

                x_axis_signal = window['x_axis'].to_numpy()
                y_axis_signal = window['y_axis'].to_numpy()
                z_axis_signal = window['z_axis'].to_numpy()

                x_axis_scaleogram = _create_scaleogram(x_axis_signal, n=scale, wavelet=wavelet)
                y_axis_scaleogram = _create_scaleogram(y_axis_signal, n=scale, wavelet=wavelet)
                z_axis_scaleogram = _create_scaleogram(z_axis_signal, n=scale, wavelet=wavelet)

                x_axis_scaleograms.append(x_axis_scaleogram)
                y_axis_scaleograms.append(y_axis_scaleogram)
                z_axis_scaleograms.append(z_axis_scaleogram)

            # Step 4: Concatenate scaleograms vertically and save the images
            output_dir = f"testing_scaleograms/user_{user_id}/{wavelet}/{scale}"
            os.makedirs(output_dir, exist_ok=True)

            for i in range(len(x_axis_scaleograms)):
                scaleogram_combined = np.concatenate(
                    (x_axis_scaleograms[i], y_axis_scaleograms[i], z_axis_scaleograms[i]),
                    axis=0
                )

                plt.imshow(scaleogram_combined, aspect='auto')
                plt.axis('off')
                plt.savefig(f"{output_dir}/scaleogram_{i}.png")
                plt.close()
