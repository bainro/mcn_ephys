import numpy as np

n = 100
true = np.zeros((n,n))

for i in range(n):
    true[i,i] = i

true_shape = true.shape
padded_shape = []
for dim in true_shape:
    padded_shape.append(dim * 2)
padded_shape = tuple(padded_shape)

dtype = 'float32'
dtype = np.dtype(dtype)
d = dict(
    descr=np.lib.format.dtype_to_descr(dtype),
    fortran_order=False,
    shape=padded_shape,
)
f = r"/tmp/test.npy"
with open(f, 'w+b') as fp:
    np.lib.format._write_array_header(fp, d, (1,0))
    offset = fp.tell()

rhd = true[:,:n//2]
s = (np.prod(padded_shape, dtype=np.int64),)
lfp = np.memmap(f, dtype=dtype, shape=s, mode='r+', offset=offset)
lfp[:rhd.flatten().shape[0]] = rhd.flatten()
del lfp

rhd = true[:,n//2:]
lfp = np.memmap(f, dtype=dtype, shape=s, mode='r+', offset=offset)
lfp[rhd.flatten().shape[0]:rhd.flatten().shape[0]*2] = rhd.flatten()
del lfp

d['shape'] = true_shape
with open(f, 'r+b') as fp:
    np.lib.format._write_array_header(fp, d, (1,0))
    
new_lfp = np.load(f)
new = new_lfp.sum()
old = true.sum()
assert new == old, f"{new} != {old} :("
print(":)")