### IMAGE CLASSIFIER USING TENSORFLOW | KERAS
A Image Classifier uses deep learning technique to classify images on what it is trained on by extracting features from images which are provided as an examples (during training). It uses CNN (Convolutional Neural Network) to extract features from images. An Image Classifier can be trained on multiple classes and is able to identify the image with perfect score.

In this repository, Tensorflow + Keras are used for Image Classfication.

-------------

### Installing packages
----

To run the code use below command to install the required packages.

```bat
pip install -r requirements.txt
```
-------------
### Dataset Structure

----

The data structure for this repo is shown as below...
```
data
└───dataset  (This is the main folder)
        └─── Class 1 folder (sub folders...)
        |    │   │   Image1.jpg
        |    │   │   Image2.jpg
        |            ...
        └─── Class 2 folder
        |    │   │   Image1.jpg
        |    │   │   Image2.jpg
        |            ...
        .
        .
        .
        └─── Class 2 folder
        |    │   │   Image1.jpg
        |    │   │   Image2.jpg
        |            ...
```

If your dataset is in this structure then you can use below command from project path to split data into train and val sets.

```bat
python split_train_val.py --data_dir <path to dataset folder> --out_train_dir <out path to train set> --out_val_dir <out path to val set>
```

By default, the train and val folders are saved under "data/training data/" folder.

Note: If you already have Train and Val Sets there is no need to run the above command.


-------------

### Train

----

Now that Training data is prepared, run below command to train your dataset

```bat
python train.py --train_path <path to train folder> --val_path <out path to val folder> --model_arch  ResNet50
```

There are many more model architectures are available like, ""Xception, ResNet50, InceptionV3, MobileNet, FineTune ""

To finetune the model, you should specify ```--model_finetune_path <path to model>```


-------------

### Test

----

To test the model, specify image_path and model_path under <b>test.py</b> script. The output provides the class name with a score.

-------------

```bat
python test.py
```

### Logs

----

The training logs are stored under "logs/fit/" directory. use below tensorboard command to review loss and accuracy for each epoch.

```bat
tensorboard --logdir logs/fit/
```



### Licence
See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
