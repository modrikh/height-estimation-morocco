from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50

def resnet_encoder(input_tensor):
    # NOTE: weights=None because inputs are not 3-band RGB
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

    skip_connections = [
        base_model.get_layer('conv1_relu').output,             # 64x64
        base_model.get_layer('conv2_block3_out').output,       # 32x32
        base_model.get_layer('conv3_block4_out').output,       # 16x16
        base_model.get_layer('conv4_block6_out').output        # 8x8
    ]

    encoder_output = base_model.get_layer('conv5_block3_out').output  # 4x4

    return skip_connections, encoder_output

def build_mbhr_resnet(input_shape_s1, input_shape_s2, input_shape_dem):
    input_s1 = Input(shape=input_shape_s1, name="s1_input")   # e.g., (128, 128, 2)
    input_s2 = Input(shape=input_shape_s2, name="s2_input")   # e.g., (128, 128, 4)
    input_dem = Input(shape=input_shape_dem, name="dem_input")  # e.g., (128, 128, 1)

    # ResNet-style encoders for S1 and S2
    skips_s1, enc_s1 = resnet_encoder(input_s1)
    skips_s2, enc_s2 = resnet_encoder(input_s2)

    # Simple CNN encoder for DEM
    x_dem = layers.Conv2D(32, 3, activation='relu', padding='same')(input_dem)
    x_dem = layers.MaxPooling2D()(x_dem)  # 128 → 64
    x_dem = layers.Conv2D(64, 3, activation='relu', padding='same')(x_dem)
    x_dem = layers.MaxPooling2D()(x_dem)  # 64 → 32
    x_dem = layers.Conv2D(128, 3, activation='relu', padding='same')(x_dem)
    x_dem = layers.MaxPooling2D()(x_dem)  # 32 → 16
    x_dem = layers.Conv2D(256, 3, activation='relu', padding='same')(x_dem)
    x_dem = layers.MaxPooling2D()(x_dem)  # 16 → 8

    # Merge all encoder outputs at bottleneck
    merged = layers.Concatenate()([enc_s1, enc_s2, x_dem])
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(merged)

    # Decoder path (U-Net)
    x = layers.UpSampling2D()(x)  # 4 → 8
    x = layers.Concatenate()([x, skips_s1[3], skips_s2[3]])
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)  # 8 → 16
    x = layers.Concatenate()([x, skips_s1[2], skips_s2[2]])
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)  # 16 → 32
    x = layers.Concatenate()([x, skips_s1[1], skips_s2[1]])
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)  # 32 → 64
    x = layers.Concatenate()([x, skips_s1[0], skips_s2[0]])
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)  # 64 → 128
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)

    output = layers.Conv2D(1, 1, activation='linear', name='regression_output')(x)

    model = Model(inputs=[input_s1, input_s2, input_dem], outputs=output, name="MBHR_ResNet")
    return model
