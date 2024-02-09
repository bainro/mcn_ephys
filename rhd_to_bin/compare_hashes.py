import os
import sys
import hashlib

def hash_files(files):
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
        return False
    return True

if __name__ == "__main__":
    files = []
    for f in sys.argv[1:]:
        files.append(os.path.join(f))
    assert hash_files(files), "Files are different :("
