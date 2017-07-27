import struct

import numpy as np

def read_idx(filename):
	with open(filename, 'rb') as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

a = read_idx('data/raw/train-images-idx3-ubyte')
print(a)
