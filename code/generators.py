import tensorflow as tf
import numpy as np
import rasterio
from pathlib import Path
from code.config import *
from code.augment import GRD_toRGB_S1, GRD_toRGB_S2, load_dem  # Use your augment functions

class Cust_DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_tuples, batch_size=16, training=False):
        """
        file_tuples: list of (label_file, s1_file, s2_file, dem_file)
        """
        self.file_tuples = file_tuples
        self.batch_size = batch_size
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.file_tuples) // self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.file_tuples)

    def __getitem__(self, idx):
        trials = 0
        while trials < 10:
            indices = np.random.choice(len(self.file_tuples), self.batch_size)
            batch_s1, batch_s2, batch_dem, batch_label = [], [], [], []

            for i in indices:
                try:
                    label_file, s1_file, s2_file, dem_file = self.file_tuples[i]

                    s1 = GRD_toRGB_S1(S1_PATH, s1_file)
                    s2 = GRD_toRGB_S2(S2_PATH, s2_file, S2_MAX)
                    dem = load_dem(DEM_PATH, dem_file)
                    label = self.read_label(label_file)

                    if np.all(label == 0) or np.std(label) < 1e-3:
                        continue

                    if self.training:
                        s1 = self.random_augment(s1)
                        s2 = self.random_augment(s2)
                        dem = self.random_augment(dem)
                        label = self.random_augment(label)

                    batch_s1.append(s1)
                    batch_s2.append(s2)
                    batch_dem.append(dem)
                    batch_label.append(label)

                except Exception as e:
                    print(f"Skipping sample due to error: {e}")
                    continue

            if batch_s1:
                return (
                    (np.array(batch_s1, dtype=np.float32),
                     np.array(batch_s2, dtype=np.float32),
                     np.array(batch_dem, dtype=np.float32)),
                    np.array(batch_label, dtype=np.float32)
                )

            trials += 1
            print("Retrying batch due to empty or invalid samples...")

        raise ValueError("Could not generate valid batch after multiple attempts.")

    def read_label(self, fname):
        label_path = LABEL_PATH / fname
        with rasterio.open(label_path) as src:
            label = src.read(1)
        label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT))
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.clip(label, 0, 100).astype(np.float32) / 100.0
        return label[..., np.newaxis]

    @staticmethod
    def random_augment(image):
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        if np.random.rand() > 0.5:
            image = np.flipud(image)
        if np.random.rand() > 0.5:
            image = image * np.random.uniform(0.9, 1.1)
        return np.clip(image, 0, 1)
