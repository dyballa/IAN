from distutils.core import setup
from distutils.extension import Extension
import numpy
cmdclass = {}
try:
	from Cython.Distutils import build_ext
	ext = 'pyx'
	cmdclass['build_ext'] = build_ext
except:
	ext = 'c'
ext_modules = [ Extension( "test_cython.cutils", ['test_cython/cutils.' + ext],
				include_dirs=[numpy.get_include()],
				) ]
setup(
    name='IAN',
    version='1.0.0',
    packages=['ian',],
    author="Luciano Dyballa",
    description="Iterated Adaptive Neighborhoods for manifold learning and dimensionality estimation.",
    license='BSD 3-Clause License',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'scipy',
        'cvxpy>=1.1.18',
        'matplotlib',
        'scikit-learn',
        #'cython>=0.29.27',
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
