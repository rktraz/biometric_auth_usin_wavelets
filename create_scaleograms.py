import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def _create_scaleogram(signal: np.ndarray, n=16, wavelet="mexh") -> np.ndarray:
    """Creates scaleogram for signal, and returns it.

    The resulting scaleogram represents scale in the first dimension, time in
    the second dimension, and the color shows amplitude.
    """

    scale_list = np.arange(start=0, stop=len(signal)) / n + 1
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]
    return scaleogram


started = time.time()

# for df, df_name in zip([df_train, df_test], ["train_dataset", "test_dataset"]):
for df, df_name in zip([df_train], ["test_dataset"]):
    print(time.strftime("%H:%M:%S", time.localtime()))
    print(f"Started processing {df_name}.")
    print(32 * "*")
    print()
    # for wavelet, n in zip(["mexh", "morl"], [16, 8]):
    wavelet = "mexh"
    n = 16
    n_users = df.user_id.nunique()
    SAVE_TO = f"{df_name}_{wavelet}_method"

    # Step 1: Create windows of size 225 with a shift factor of 30
    window_size = 225
    shift_factor = 30

    # Step 2: Perform operations separately for each user
    users = df['user_id'].unique()

    for user_id in users:
        user_data = df[df['user_id'] == user_id]

        # Check if folder already exists for user_id
        output_dir = f"{SAVE_TO}/user_{user_id}"
        if os.path.exists(output_dir):
            continue
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Step 3: Create scaleograms for each axis for every window
        x_axis_scaleograms = []
        y_axis_scaleograms = []
        z_axis_scaleograms = []

        for i in range(0, len(user_data) - window_size + 1, shift_factor):
            window = user_data.iloc[i:i+window_size]

            x_axis_signal = window['x_axis'].to_numpy()
            y_axis_signal = window['y_axis'].to_numpy()
            z_axis_signal = window['z_axis'].to_numpy()

            x_axis_scaleogram = _create_scaleogram(x_axis_signal, n=n, wavelet=wavelet)
            y_axis_scaleogram = _create_scaleogram(y_axis_signal, n=n, wavelet=wavelet)
            z_axis_scaleogram = _create_scaleogram(z_axis_signal, n=n, wavelet=wavelet)

            x_axis_scaleograms.append(x_axis_scaleogram)
            y_axis_scaleograms.append(y_axis_scaleogram)
            z_axis_scaleograms.append(z_axis_scaleogram)

        # Step 4: Concatenate scaleograms vertically and save the images
        for i in range(len(x_axis_scaleograms)):
            scaleogram_combined = np.concatenate(
                (x_axis_scaleograms[i], y_axis_scaleograms[i], z_axis_scaleograms[i]),
                 axis=0
            )
            plt.imshow(scaleogram_combined, aspect='auto')
            plt.axis('off')
            plt.savefig(f"{output_dir}/scaleogram_{i}.png")
            plt.close()
        print()
        print("Processed user with id", user_id)
    print(f"Processing of {SAVE_TO} folder is done. It took {time.time() - started} sec.\n")
    print(32 * "*")
    print()