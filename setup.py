from setuptools import setup, Extension
import numpy
import tslearn_addons

have_cython = False
try:
    from Cython.Distutils import build_ext as _build_ext
    have_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext

list_pyx = ['cylrdtw']
if have_cython:
    ext = [Extension('tslearn_addons.%s' % s, ['tslearn_addons/%s.pyx' % s]) for s in list_pyx]
else:
    ext = [Extension('tslearn_addons.%s' % s, ['tslearn_addons/%s.c' % s]) for s in list_pyx]

setup(
    name="tslearn_addons",
    description="Add-ons to the tslearn toolkit",
    include_dirs=[numpy.get_include()],
    packages=['tslearn_addons'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'tslearn'],
    ext_modules=ext,
    cmdclass={'build_ext': _build_ext},
    version=tslearn_addons.__version__,
    url="http://tslearn_addons.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)  # TODO: test package_data option on PyPI deployment
