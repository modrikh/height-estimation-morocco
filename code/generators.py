# code/generators.py

import tensorflow as tf
import cv2
import numpy as np
import rasterio
from code.config import *
from code.augment import GRD_toRGB_S1, GRD_toRGB_S2, load_dem

class MultiTaskPatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_tuples, batch_size, patch_size, training=False, **kwargs):
        super().__init__(**kwargs)
        self.file_tuples = file_tuples
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        return len(self.file_tuples) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.file_tuples)

    def __getitem__(self, idx):
        batch_files = self.file_tuples[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_s1, batch_s2, batch_dem, batch_labels_reg, batch_labels_cls = [], [], [], [], []

        for label_fname, s1_fname, s2_fname, dem_fname in batch_files:
            try:
                s1_full = GRD_toRGB_S1(S1_PATH, s1_fname)
                s2_full = GRD_toRGB_S2(S2_PATH, s2_fname, S2_MAX)
                dem_full = load_dem(DEM_PATH, dem_fname)
                worldcover_map = self.read_worldcover_map(label_fname)
                height_label_full = self.create_height_label_from_map(worldcover_map)
                class_label_full = self.create_class_label_from_map(worldcover_map)

                h, w, _ = s1_full.shape
                if h < self.patch_size or w < self.patch_size: continue
                x, y = np.random.randint(0, w - self.patch_size), np.random.randint(0, h - self.patch_size)

                s1_patch, s2_patch, dem_patch, height_patch, class_patch = [
                    img[y:y+self.patch_size, x:x+self.patch_size]
                    for img in [s1_full, s2_full, dem_full, height_label_full, class_label_full]
                ]

                # (Logique d'augmentation ici)
                
                batch_s1.append(s1_patch); batch_s2.append(s2_patch); batch_dem.append(dem_patch)
                batch_labels_reg.append(height_patch); batch_labels_cls.append(class_patch)
            except Exception as e:
                print(f"Avertissement : Erreur sur le patch pour {s1_fname}. Erreur: {e}")
                continue
        
        if not batch_s1: return self.__getitem__((idx + 1) % len(self))
        
        return (
            (np.array(batch_s1), np.array(batch_s2), np.array(batch_dem)),
            {"output_height": np.array(batch_labels_reg), "output_segmentation": np.array(batch_labels_cls)}
        )

    def read_worldcover_map(self, fname):
        label_path = LABEL_PATH / fname
        with rasterio.open(label_path) as src: label = src.read(1)
        return cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

    def create_height_label_from_map(self, worldcover_map):
        height_map = np.where(worldcover_map == BUILT_UP_CLASS_ID, DEFAULT_BUILDING_HEIGHT, 0)
        normalized_height = np.clip(height_map, 0, 100).astype(np.float32) / 100.0
        return normalized_height[..., np.newaxis]

    def create_class_label_from_map(self, worldcover_map):
        binary_mask = np.where(worldcover_map == BUILT_UP_CLASS_ID, 1, 0).astype(np.int32)
        return tf.keras.utils.to_categorical(binary_mask, num_classes=NUM_CLASSES).astype(np.float32)