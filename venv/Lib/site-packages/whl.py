#!/usr/bin/env python
import argparse
import base64
import hashlib
import os
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("-2", dest="py2", action="store_true")
parser.add_argument("-3", dest="py3", action="store_true")
args = parser.parse_args()
py2 = args.py2
py3 = args.py3
if not (py2 or py3):
    # universal by default
    py2 = py3 = True
assert py2 or py3

here = os.path.dirname(os.path.abspath(__file__))
name = os.path.basename(here)
version = "0.0.1"

# https://packaging.python.org/specifications/core-metadata/
METADATA = """\
Metadata-Version: 2.1
Name: {}
Version: {}

Minimalist wheel building
""".format(name, version)

# https://www.python.org/dev/peps/pep-0427/
WHEEL = """\
Wheel-Version: 1.0
Generator: whl (0.1)
Root-Is-Purelib: true
"""
if py2:
    WHEEL += "Tag: py2-none-any\n"
if py3:
    WHEEL += "Tag: py3-none-any\n"

blacklist = {"README.rst", "setup.py", "setup.cfg"}
dist_files = []
for root, dirs, fnames in os.walk("."):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for fname in fnames:
        if fname not in blacklist and not fname.endswith((".whl", ".pyc")):
            relpath = os.path.join(root, fname)
            dist_files.append(relpath)

tags = "py2.py3"
if py2 and not py3:
    tags = "py2"
elif py3 and not py2:
    tags = "py3"

whl_name = "{}-{}-{}-none-any.whl".format(name.replace("-", "_"), version, tags)
dist_info_name = "{}-{}.dist-info".format(name.replace("-", "_"), version)

def get_record(path, data=None):
    if data is None:
        with open(path, "rb") as f:
            data = f.read()
    checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=')
    if path.startswith("./"):
        path = path[2:]
    line = "{},sha256={},{}".format(path, checksum.decode(), len(data))
    return line

RECORD = []
zf = zipfile.ZipFile(whl_name, "w")
for path in dist_files:
    zf.write(path)
    record = get_record(path)
    RECORD.append(record)
    print(record)

info_metadata = os.path.join(dist_info_name, "METADATA")
RECORD.append(get_record(info_metadata, data=METADATA.encode()))
zf.writestr(info_metadata, METADATA)

info_wheel = os.path.join(dist_info_name, "WHEEL")
RECORD.append(get_record(info_wheel, data=WHEEL.encode()))
zf.writestr(info_wheel, WHEEL)

RECORD.append("{},,".format(os.path.join(dist_info_name, "RECORD")))
zf.writestr(os.path.join(dist_info_name, "RECORD"), "\n".join(RECORD))

zf.close()
