from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bnmorph',
    ext_modules=[
        CUDAExtension('bnmorph_getcorpts', [
            'bnmorph_getcorpts.cpp',
            'bnmorph_getcorpts_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
