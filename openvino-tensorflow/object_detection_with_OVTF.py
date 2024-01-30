import os
import pathlib

import tensorflow as tf

import openvino_tensorflow as ovtf

import argparse

if __name__ == "__main__":

    backend_name = "CPU"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        help="Optional. Specify the target device to infer on; "
        "CPU, GPU, MYRIAD, or VAD-M is acceptable. Default value is CPU.")

    parser.add_argument(
        "--disable_ovtf",
        help="Optional."
        "Disable openvino_tensorflow pass and run on stock TF.",
        action='store_true')

    args = parser.parse_args()

    if args.backend:
        backend_name = args.backend

    if not args.disable_ovtf:
        # Print list of available backends
        print('Available Backends:')
        backends_list = ovtf.list_backends()
        for backend in backends_list:
            print(backend)
        ovtf.set_backend(backend_name)
    else:
        ovtf.disable()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# GPU 設定為固定為 2GB
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])
#

# 下載模型，並解壓縮
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    # 解壓縮
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)


MODEL_DATE = '20200711'
MODEL_NAME = 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
print(PATH_TO_MODEL_DIR)

# 讀取 PATH_TO_MODEL_DIR 目錄下所有目錄及檔案
from os import listdir

for f in listdir(PATH_TO_MODEL_DIR):
    print(f)

# 從下載的目錄載入模型，耗時甚久
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('載入模型...', end='')
start_time = time.time()

# 載入模型
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# tf.keras.utils.plot_model(detect_fn, to_file='model.png')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'共花費 {elapsed_time} 秒.')


## 建立 Label 的對照表


# 下載 labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)


LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)
print(PATH_TO_LABELS)

# 建立 Label 的對照表 (代碼與名稱)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

## 選一張圖片進行物件偵測

# 選一張圖片(./images_2/image2.jpg)進行物件偵測
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 不顯示警告訊息
import warnings

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

# 開啟一張圖片
# image_np = np.array(Image.open('./images_2/image2.jpg'))
image_np = np.array(Image.open('./COCO_train2014_000000000049.jpg'))
# image_np = np.array(Image.open('./train2014/COCO_train2014_000000173610.jpg'))

# 轉為 TensorFlow tensor 資料型態
input_tensor = tf.convert_to_tensor(image_np)
# 加一維，變為 (筆數, 寬, 高, 顏色)
input_tensor = input_tensor[tf.newaxis, ...]
# 這方法也可以
# input_tensor = np.expand_dims(image_np, 0)

detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))

# detections：物件資訊 內含 (候選框, 類別, 機率)
print(f'物件個數：{num_detections}')
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}

detections['num_detections'] = num_detections
# 轉為整數
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

print(f'物件資訊 (候選框, 類別, 機率)：')
for detection_boxes, detection_classes, detection_scores in \
        zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
    print(np.around(detection_boxes, 4), detection_classes, round(detection_scores * 100, 2))


#框篩選，並將圖片的物件加框


image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.40, # 繪製最低機率
    agnostic_mode=False)

plt.figure(figsize=(15, 10))
plt.imshow(image_np_with_detections, cmap='viridis')

plt.savefig('./detection.png', dpi=300)
# plt.show()
from IPython.display import Image
Image('./detection.png')

print()
print("end")
