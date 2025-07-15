
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50

def build_multitask_mbhr_resnet(input_shape_s1, input_shape_s2, input_shape_dem, num_classes):
    """
    Construit le modèle multi-tâches en utilisant un backbone ResNet50 PARTAGÉ.
    """
    # === Entrées ===
    input_s1 = Input(shape=input_shape_s1, name="input_s1")
    input_s2 = Input(shape=input_shape_s2, name="input_s2")
    input_dem = Input(shape=input_shape_dem, name="input_dem")
    inputs = [input_s1, input_s2, input_dem]

    # --- Pré-traitement pour ResNet50 (3 canaux) ---
    s1_pre = layers.Conv2D(3, kernel_size=1, padding='same', name='s1_to3ch')(input_s1)
    s2_pre = layers.Conv2D(3, kernel_size=1, padding='same', name='s2_to3ch')(input_s2)

    # === Encodeur ResNet50 PARTAGÉ ===
    resnet_backbone = ResNet50(include_top=False, weights=None, input_shape=(None, None, 3))
    
    layer_names = [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out' 
    ]
    feature_extractor = Model(inputs=resnet_backbone.input, outputs=[resnet_backbone.get_layer(name).output for name in layer_names], name="resnet_feature_extractor")
    
    s1_skips_and_bottleneck = feature_extractor(s1_pre)
    s2_skips_and_bottleneck = feature_extractor(s2_pre)
    
    s1_skips = s1_skips_and_bottleneck[:-1]
    s1_bottleneck = s1_skips_and_bottleneck[-1]
    
    s2_skips = s2_skips_and_bottleneck[:-1]
    s2_bottleneck = s2_skips_and_bottleneck[-1]


    # === Branche DEM Simple ===
    x_dem = layers.Conv2D(64, 3, activation='relu', padding='same')(input_dem); x_dem = layers.MaxPooling2D()(x_dem); dem_skip1 = x_dem
    x_dem = layers.Conv2D(128, 3, activation='relu', padding='same')(x_dem); x_dem = layers.MaxPooling2D()(x_dem); dem_skip2 = x_dem
    x_dem = layers.Conv2D(256, 3, activation='relu', padding='same')(x_dem); x_dem = layers.MaxPooling2D()(x_dem); dem_skip3 = x_dem
    x_dem = layers.Conv2D(512, 3, activation='relu', padding='same')(x_dem); x_dem = layers.MaxPooling2D()(x_dem); dem_skip4 = x_dem
    x_dem = layers.Conv2D(1024, 3, activation='relu', padding='same')(x_dem); x_dem = layers.MaxPooling2D()(x_dem); dem_bottleneck = x_dem
    

    # === Fusion au Bottleneck ===
    merged_bottleneck = layers.Concatenate()([s1_bottleneck, s2_bottleneck, dem_bottleneck])
    x = layers.Conv2D(1024, 3, padding='same', activation='relu', name="fusion_bottleneck_conv")(merged_bottleneck)

    # === Décodeur (U-Net style) ===
    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, s1_skips[3], s2_skips[3], dem_skip4]); x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, s1_skips[2], s2_skips[2], dem_skip3]); x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, s1_skips[1], s2_skips[1], dem_skip2]); x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, s1_skips[0], s2_skips[0], dem_skip1]); x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    final_decoder_features = layers.UpSampling2D()(x)
    
    # === TÊTES DE PRÉDICTION MULTI-TÂCHES ===
    reg_branch = layers.Conv2D(32, 3, padding='same', activation='relu', name='reg_head_conv')(final_decoder_features)
    output_regression = layers.Conv2D(1, 1, activation='sigmoid', name='output_height')(reg_branch)

    cls_branch = layers.Conv2D(32, 3, padding='same', activation='relu', name='cls_head_conv')(final_decoder_features)
    output_classification = layers.Conv2D(num_classes, 1, activation='softmax', name='output_segmentation')(cls_branch)

    # === Modèle Final avec 2 Sorties ===
    model = Model(
        inputs=[input_s1, input_s2, input_dem], 
        outputs=[output_regression, output_classification], 
        name="MultiTask_MBHR_ResNet_SharedBackbone"
    )
    return model