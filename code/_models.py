from _unet_xception import UnetXception
from _unet_resnet import UnetResnet50

import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=0.005):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)