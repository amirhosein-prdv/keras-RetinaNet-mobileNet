import cv2
import keras
import pandas as pd
from keras.models import load_model
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.models.mobilenet import custom_objects
setup_gpu('0')

classes = "D:\Amirhosein\Object_Detection\\tag-detection-retinanet_OtherNet\Code\dataset\\classes.csv"
val_path = "D:\Amirhosein\Object_Detection\\tag-detection-retinanet_OtherNet\Code\dataset\\val.csv"
test_path = "D:\Amirhosein\Object_Detection\\tag-detection-retinanet_OtherNet\Code\dataset\\test.csv"
model_path = "D:\Amirhosein\Object_Detection\keras-retinanet-mobilenet\snapshots\mobilenet_a1_s8_rdc2\mobilenet_a1_s8_rdc2_best.h5"

test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for testing data
test_generator = CSVGenerator(
    csv_data_file=test_path,
    csv_class_file=classes,
    image_data_generator=test_image_data_generator,
    batch_size=2,
    )

model = load_model(model_path, custom_objects=custom_objects)

print(model.summary())

logs = evaluate(
    generator=test_generator,
    model=model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path='output')