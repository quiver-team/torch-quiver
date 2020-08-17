#!/usr/bin/env python3

import os
import sysconfig

import torch
from torch.utils import cpp_extension


def gen_cmake_config():
    include_dirs = [
        os.path.dirname(sysconfig.get_config_h_filename()),
    ]
    include_dirs += cpp_extension.include_paths()

    library_dirs = []
    library_dirs += cpp_extension.library_paths()

    for p in include_dirs:
        print('INCLUDE_DIRECTORIES(SYSTEM %s)' % (p))

    for p in library_dirs:
        print('LINK_DIRECTORIES(%s)' % (p))

    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    print('ADD_DEFINITIONS(-D%s=%s)' % ('_GLIBCXX_USE_CXX11_ABI', abi))

    if not torch.cuda.is_available():
        print('ADD_DEFINITIONS(-D%s=%s)' % ('NO_CUDA', '1'))


gen_cmake_config()
