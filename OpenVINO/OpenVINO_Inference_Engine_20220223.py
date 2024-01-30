# https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_Integrate_with_customer_application_new_API.html?sw_type=switcher-python

# Integrate Inference Engine

print( "Import Inference Module..." )

from openvino.inference_engine import IECore

print( "Create Inference Engine Core..." )

ie = IECore()

print( "Read model..." )

net = ie.read_network(model="/home/ytl/Downloads/1xx3/saved_model.xml", weights="/home/ytl/Downloads/1xx3/saved_model.bin")  # Returns An IENetwork object

print( "Configure Input and Output of the Model..." )

inputs = net.input_info
print(inputs)  # {'input': <openvino.inference_engine.ie_api.InputInfoPtr object at 0x7fb78e630600>}

input_name = next(iter(net.input_info))
print( input_name )  # input
print( net.inputs[input_name].shape )  # [1, 3, 224, 224]

outputs = net.outputs
print( outputs )  # {'MobilenetV2/Predictions/Reshape_1': <openvino.inference_engine.ie_api.DataPtr object at 0x7fc644b008b0>}

output_name = next(iter(net.outputs))
print( output_name )  # MobilenetV2/Predictions/Reshape_1


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

print( "Prepare input..." )

import cv2
import numpy as np

image = cv2.imread("/home/ytl/Downloads/5.jpeg")

cv2.imshow("iuput", image)

#cv2.waitKey(0)

print(image.shape)  # (183, 275, 3)

image = cv2.resize(image, (224, 224))

cv2.imshow("resize", image)

#cv2.waitKey(0)

print( "Test input..." )

print(image.shape)  # (224, 224, 3)

print(image)

# Resize with OpenCV your image if needed to match with net input shape
# N, C, H, W = net.input_info[input_name].tensor_desc.dims
# image = cv2.resize(src=image, dsize=(W, H))

print( "Converting image to NCHW format with FP32 type..." )

print(type(image))  # <class 'numpy.ndarray'>

input_data = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32)

print(type(input_data))  # <class 'numpy.ndarray'>

print( input_data )  # above are the same.

print( "Start Inference..." )

result = exec_net.infer({input_name: input_data})
print( result )  # {'MobilenetV2/Predictions/Reshape_1': array([[8.7738597e-05, 3.0000857e-04, 8.1778628e-05, ..., 5.0595463e-05, 1.1311602e-04, 3.4677156e-05]], dtype=float32)}

print( type(result) )  # <class 'dict'>

print( "Print all keys in dict..." )

for key, value in result.items() :
    print (key, value)

print( "Process the Inference Results..." )

output = result[output_name]

print( type( output[0] ) )  # <class 'numpy.ndarray'>

idx = np.argsort(np.squeeze(result[output_name][0]))[::-1]
print(idx)  # [488 591 762 ... 962 660 499]

print( "Process the Results..." )

for i in range(5):
    print(idx[i]+1, result[output_name][0][idx[i]])































