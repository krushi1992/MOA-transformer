# --------------------------------------------------------
# Adopted from Swin Transformer
# Modified by Krushi Patel
# --------------------------------------------------------

from .MOA_transformer import MOATransformer


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'MOA':
        model = MOATransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.MOA.PATCH_SIZE,
                                in_chans=config.MODEL.MOA.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.MOA.EMBED_DIM,
                                depths=config.MODEL.MOA.DEPTHS,
                                num_heads=config.MODEL.MOA.NUM_HEADS,
                                window_size=config.MODEL.MOA.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.MOA.MLP_RATIO,
                                qkv_bias=config.MODEL.MOA.QKV_BIAS,
                                qk_scale=config.MODEL.MOA.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.MOA.APE,
                                patch_norm=config.MODEL.MOA.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
