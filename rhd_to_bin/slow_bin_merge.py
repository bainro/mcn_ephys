import os
import sys

path1 = os.path.join(sys.argv[1])
path2 = os.path.join(sys.argv[2])

with open(path1, "ab") as file1, open(path2, "rb") as file2:
    file1.write(file2.read())