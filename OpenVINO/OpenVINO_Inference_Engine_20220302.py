from openvino.inference_engine import IECore

ie = IECore()

net = ie.read_network(model="/home/ytl/Downloads/1xx3/saved_model.xml", weights="/home/ytl/Downloads/1xx3/saved_model.bin")  # Returns An IENetwork object

inputs = net.input_info

input_name = next(iter(net.input_info))

outputs = net.outputs

output_name = next(iter(net.outputs))

print("Inputs:")

for name, info in net.input_info.items():
    print("\tname: {}".format(name))
    print("\tshape: {}".format(info.tensor_desc.dims))
    print("\tlayout: {}".format(info.layout))
    print("\tprecision: {}\n".format(info.precision))
    

print("Outputs:")

for name, info in net.outputs.items():
    print("\tname: {}".format(name))
    print("\tshape: {}".format(info.shape))
    print("\tlayout: {}".format(info.layout))
    print("\tprecision: {}\n".format(info.precision))
    
exec_net = ie.load_network(network=net, device_name="CPU")  # Returns An ExecutableNetwork object

# ================================================================================

import numpy as np

import tensorflow as tf

image_path="/home/ytl/Downloads/5.jpeg"

image = tf.io.read_file(image_path)

image = tf.image.decode_jpeg(image, channels=3)

#print(image.shape)  # (183, 275, 3)

image = tf.image.resize(image, (224, 224))

#print(image.shape)  # (224, 224, 3)

image = tf.keras.applications.resnet.preprocess_input(image)  # Resnet

#print(image.shape)  # (224, 224, 3)

input_data = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32)

#print(input_data.shape)  # (1, 3, 224, 224)

# ================================================================================

result = exec_net.infer({input_name: input_data})

#print(result[list(result.keys())[0]].shape)  # (1, 2048, 7, 7)

RNN_input = np.transpose(result[list(result.keys())[0]], (0, 2, 3, 1)).astype(np.float32)

#print(RNN_input.shape)  # (1, 7, 7, 2048)

np.save("/home/ytl/0302.npy", RNN_input)


























