import argparse
import pdb
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from net import model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===============================================================================
# Define Hyper parameters

"""Read the arguments of the program."""
arg_parse = argparse.ArgumentParser()

arg_parse.add_argument("--model_arch", required=False,
                       help="Available options-[Xception, ResNet50, InceptionV3, MobileNet, FineTune]",
                       default="ResNet50", type=str)

arg_parse.add_argument("--model_finetune_path", required=False,
                       help="Specify model path to finetune the model. eg:'checkpoints/ABC.h5'", default=None, type=str)

arg_parse.add_argument("--train_path", required=False,
                       help="Specify train dir", default='data/training_data/train', type=str)

arg_parse.add_argument("--val_path", required=False,
                       help="Specify val dir", default='data/training_data/val', type=str)

arg_parse.add_argument("--model_out_dir", required=False,
                       help="Specify where to save model checkpoints", default='./checkpoints/', type=str)

arg_parse.add_argument("--learning_rate", required=False,
                       help="Specify learning rate", default=0.0001, type=float)

arg_parse.add_argument("--batch_size", required=False,
                       help="Specify Batch Size", default=12, type=int)

arg_parse.add_argument("--num_epochs", required=False,
                       help="Specify number of epochs", default=20, type=int)

arg_parse.add_argument("--image_width", required=False,
                       help="Specify Input Image Width", default=256, type=int)

arg_parse.add_argument("--image_height", required=False,
                       help="Specify Input Image height", default=256, type=int)

arg_parse.add_argument("--no_transfer_learning", required=False,
                       help="add this arg. to prevent using transfer learning", action='store_false')

args = arg_parse.parse_args()

TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
IMAGE_DIMS = (args.image_width, args.image_height, 3)
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
MODEL_ARCH = args.model_arch
FINETUNEPATH = args.model_finetune_path
TRANSFER_LEARNING = args.no_transfer_learning
CLASS_LABELS = sorted(os.listdir(TRAIN_PATH))

MODEL_SAVE_DIR = args.model_out_dir
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===============================================================================

if len(CLASS_LABELS) != len(os.listdir(VAL_PATH)):  # Pre-checking no. of classes before proceeding any further...
    raise Exception("Number of classes in train data and val data are not equal!")

# Load Model Architecture
model_final = model(MODEL_ARCH, FINETUNEPATH, IMAGE_DIMS, transfer_learning=TRANSFER_LEARNING)

# Generate train and test data
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=15)

train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                                                    target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    class_mode="categorical")

val_datagen = ImageDataGenerator(
    rescale=1. / 255)  # No need of data aug. on val. set(Can add augs same as train_datagen if needed).

validation_generator = val_datagen.flow_from_directory(VAL_PATH,
                                                       target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]),
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       class_mode="categorical")
if MODEL_ARCH != "FineTune":
    # Adding custom Layers
    new_custom_layers = model_final.output
    new_custom_layers = Flatten()(new_custom_layers)
    new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
    new_custom_layers = Dropout(0.5)(new_custom_layers)
    new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)

    try:
        num_classes = train_generator.num_class
    except:
        num_classes = train_generator.num_classes

    predictions = Dense(num_classes, activation="softmax")(new_custom_layers)

    # creating the final model
    model_final = Model(inputs=model_final.input, outputs=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy",
                        optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum=0.9),
                        # optimizer=optimizers.RMSprop(lr=LEARNING_RATE, momentum=0.9),  # Select Best Optimizer for your data
                        # optimizer=optimizers.Adam(lr=LEARNING_RATE),
                        metrics=["accuracy"])

# select .h5 filename
if FINETUNEPATH is not None:
    file_name = MODEL_ARCH + "_"
elif TRANSFER_LEARNING:
    file_name = MODEL_ARCH + '_transfer_learning_'
else:
    file_name = MODEL_ARCH + '_without_transfer_learning_'

# Save the model according to the conditions
checkpoint = ModelCheckpoint(os.path.join(MODEL_SAVE_DIR, file_name + "{epoch:03d}" + ".h5"), monitor='val_acc',
                             verbose=1, save_best_only=False, save_weights_only=False, save_freq='epoch',
                             mode='auto', period=1)

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=0.00001)
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [earlystop, learning_rate_reduction, checkpoint, tensorboard_callback]

# Train the model
model_final.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE)

print("Train Complete! logs saved at logs/ dir.")
