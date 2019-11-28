# -*- coding: utf-8 -*-
# @Author: Liu Shaoweihua
# @Date:   2019-11-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import codecs
import shutil
import zipfile
import requests


__all__ = [
    "wget", "unzip", "rm", "mkdir", "rmdir", "mv"
]


_CURRENT_FILE = os.path.dirname(__file__)


def wget(url, save_path=None, rename=None):
    current_path = os.getcwd()
    file_name = url[url.rfind("/")+1:]
    if not save_path:
        save_path = current_path
    if not rename:
        rename = file_name
    save_path = os.path.abspath(os.path.join(save_path, rename))
    print("[wget]   downloading from {}".format(url))
    start = time.time()
    size = 0
    response = requests.get(url, stream=True)
    chunk_size = 10240
    content_size = int(response.headers["content-length"])
    if response.status_code == 200:
        print("[wget]   file size: %.2f MB" %(content_size / 1024 / 1024))
        with codecs.open(save_path, "wb") as f:
            for data in response.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data)
                print("\r"+"[wget]   %s%.2f%%"
                      %(">"*int(size*50/content_size), float(size/content_size*100)), end="")
    end = time.time()
    print("\n"+"[wget]   complete! cost: %.2fs."%(end-start))
    print("[wget]   save at: %s" %save_path)
    return save_path


def unzip(file_path, save_path=None):
    if not save_path:
        save_path = os.path.abspath("/".join(os.path.abspath(file_path).split("/")[:-1]))
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(save_path)
    print("[unzip]  file path: {}, save at {}".format(file_path, save_path))
    return save_path


def rm(file_path):
    file_path = os.path.abspath(file_path)
    os.remove(file_path)
    print("[remove] file path {}".format(file_path))
    return


def mkdir(file_path):
    file_path = os.path.abspath(file_path)
    os.makedirs(file_path)
    print("[mkdir]  create directory {}".format(file_path))
    return file_path


def rmdir(file_path):
    file_path = os.path.abspath(file_path)
    shutil.rmtree(file_path)
    print("[rmdir]  remove directory {}".format(file_path))
    return


def mv(from_file_path, to_file_path):
    from_file_path = os.path.abspath(from_file_path)
    to_file_path = os.path.abspath(to_file_path)
    os.rename(from_file_path, to_file_path)
    print("[move]   move file from {} to {}".format(from_file_path, to_file_path))
    return