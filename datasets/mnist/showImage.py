import sys
import numpy as np

image_size = 28
niWidth = 10
niHeight = 10
numImages = 10

np.set_printoptions(threshold=sys.maxsize)

f = open("train-images.idx3-ubyte", 'rb')

headerBuf = f.read(4*4)
hdt = np.dtype(np.int32)
hdt = hdt.newbyteorder(">")
header = np.frombuffer(headerBuf, dtype=hdt)
print(header)

ddt = np.dtype(np.uint8)
ddt = ddt.newbyteorder(">")
buf = f.read(image_size * image_size * numImages)
data = np.frombuffer(buf, dtype=ddt).astype(np.uint8)
# data = data.reshape(numImages, image_size*image_size)

image1 = data[0]

from PIL import Image
pilimage = Image.frombuffer(
    'L', 
    (image_size, image_size*niHeight),
    data,
    "raw",
    'L',
    0, 1)
pilimage.show()