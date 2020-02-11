import sys
import numpy as np

numLabels = 50

np.set_printoptions(threshold=sys.maxsize)

f = open("train-labels.idx1-ubyte", 'rb')

headerBuf = f.read(2*4)
hdt = np.dtype(np.int32)
hdt = hdt.newbyteorder(">")
header = np.frombuffer(headerBuf, dtype=hdt)
print(header)

ddt = np.dtype(np.uint8)
ddt = ddt.newbyteorder(">")
buf = f.read(numLabels)
data = np.frombuffer(buf, dtype=ddt).astype(np.uint8)
# data = data.reshape(numImages, image_size*image_size)
print(data)
