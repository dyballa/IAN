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
ext_modules = [ Extension( "ian.cutils", ['ian/cutils.' + ext],
				include_dirs=[numpy.get_include()],
				) ]
setup(
    name='IAN',
    version='1.1.1',
    packages=['ian',],
    author="Luciano Dyballa",
    description="Iterated Adaptive Neighborhoods for manifold learning and dimensionality estimation.",
    license='BSD 3-Clause License',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy>=1.17.3',
        'scipy>=1.3.2',
        'cvxpy>=1.1.18',
        'matplotlib>=3.1.2',
        'scikit-learn>=1.1.3',
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
