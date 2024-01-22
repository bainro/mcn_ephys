import os
import sys
import hashlib

files = []
for f in sys.argv[1:]:
    files.append(os.path.join(f))

digests = []
for filename in files:
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
        a = hasher.hexdigest()
        digests.append(a)
        print(a)

for i, d1 in enumerate(digests[:-1]):
    d2 = digests[i+1]
    f1 = files[i]
    f2 = files[i+1]
    assert d1 == d2, f"hashes for {f1} & {f2} differ :("

print("congrats! all files are identical :)")