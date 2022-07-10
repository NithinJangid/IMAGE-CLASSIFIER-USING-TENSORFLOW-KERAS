from tensorflow.keras import applications
from tensorflow.keras.models import load_model


def model(model_arch='ResNet50', model_path=None, inp_shape=(256, 256, 3), transfer_learning=True):
    if model_arch == "FineTune":
        print("FineTuning...")
    elif transfer_learning:
        print("Using Imagenet weights for transfer learning...")
        weights = "imagenet"
    else:
        print("Training without transfer learning...")
        weights = None

    if model_arch == 'Xception':
        model = applications.Xception(
            weights=weights, include_top=False, input_shape=inp_shape)
    elif model_arch == 'VGG16':
        model = applications.VGG16(
            weights=weights, include_top=False, input_shape=inp_shape)
    elif model_arch == 'VGG19':
        model = applications.VGG19(
            weights=weights, include_top=False, input_shape=inp_shape)
    elif model_arch == 'ResNet50':
        model = applications.ResNet50(
            weights=weights, include_top=False, input_shape=inp_shape)
    elif model_arch == 'InceptionV3':
        model = applications.InceptionV3(
            weights=weights, include_top=False, input_shape=inp_shape)
    elif model_arch == 'MobileNet':
        model = applications.MobileNetV2(
            weights=weights, include_top=False, input_shape=inp_shape)

    elif model_arch == 'FineTune' and model_path is not None:
        model = load_model(model_path)
        if model_path is None:
            raise Exception("No model path is defined!")

    if model_arch is None:
        raise Exception("No model architecture selected!")

    return model
