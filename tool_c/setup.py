# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
# python setup.py build_ext --inplace
setup(
name='test',
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize(Extension(
    'utils_c',
    sources=['utils_c.pyx','hello.c'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))