import glob
import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension


def find_cuda():
    # TODO: find cuda
    home = os.getenv("CUDA_HOME")
    path = os.getenv("CUDA_PATH")
    if home is not None:
        return home
    elif path is not None:
        return path
    else:
        return '/usr/local/cuda'


def have_cuda():
    if os.getenv('QUIVER_ENABLE_CUDA') == '1': return True
    import torch
    return torch.cuda.is_available()


def create_extension(with_cuda=False):
    print('Building torch_quiver with CUDA:', with_cuda)
    srcs = []
    srcs += glob.glob('srcs/cpp/src/quiver/*.cpp')
    srcs += glob.glob('srcs/cpp/src/quiver/cpu/*.cpp')
    srcs += glob.glob('srcs/cpp/src/quiver/torch/*.cpp')

    include_dirs = [
        os.path.join(os.getcwd(), 'srcs/cpp/include')
    ]
    print(include_dirs)
    
    library_dirs = []
    libraries = []
    extra_cxx_flags = [
        '-std=c++17',
        # TODO: enforce strict build
        # '-Wall',
        # '-Werror',
        # '-Wfatal-errors',
    ]
    if with_cuda:
        cuda_home = find_cuda()
        include_dirs += [os.path.join(cuda_home, 'include')]
        library_dirs += [os.path.join(cuda_home, 'lib64')]
        srcs += glob.glob('srcs/cpp/src/quiver/cuda/*.cpp')
        srcs += glob.glob('srcs/cpp/src/quiver/cuda/*.cu')
        extra_cxx_flags += ['-DHAVE_CUDA']

    if os.getenv('QUIVER_ENABLE_TRACE'):
        extra_cxx_flags += ['-DQUIVER_ENABLE_TRACE=1']

    return cpp_extension.CppExtension(
        'torch_quiver',
        srcs,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        # with_cuda=with_cuda,
        extra_compile_args={
            'cxx': extra_cxx_flags,
            'nvcc': ['-O3', '--expt-extended-lambda', '-lnuma'],
        },
    )


package_dir = './srcs/python'

setup(
    name='torch_quiver',
    version='0.1.0',
    author='Zeyuan Tan',
    author_email='zeyuan.tan@ed.ac.uk',
    url='https://github.com/quiver-team/torch-quiver',
    description=('Distributed Graph Learning Library for Pytorch Geometric'),
    keywords=['pytorch', 'distributed', 'graph'],
    license='Apache',
    python_requires='>=3.6',
    package_dir={
        '': package_dir,
    },
    packages=find_packages(package_dir),
    ext_modules=[
        create_extension(have_cuda()),
    ],
    cmdclass={
        # FIXME: parallel build, (pip_install took 1m16s)
        'build_ext': cpp_extension.BuildExtension,
    },
)
