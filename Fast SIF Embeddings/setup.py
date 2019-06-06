#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run with:

sudo python ./setup.py install
"""

import os
import sys
import warnings
import ez_setup
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class custom_build_ext(build_ext):
    """Allow C extension building to fail.

    The C extension speeds up word2vec and doc2vec training, but is not essential.
    """

    warning_message = """
********************************************************************
WARNING: %s could not be compiled. %s

Here are some hints for popular operating systems:

If you are seeing this message on Linux you probably need to
install GCC and/or the Python development package for your
version of Python.

Debian and Ubuntu users should issue the following command:

    $ sudo apt-get install build-essential python-dev

RedHat, CentOS, and Fedora users should issue the following command:

    $ sudo yum install gcc python-devel

If you are seeing this message on OSX please read the documentation
here:

http://api.mongodb.org/python/current/installation.html#osx
********************************************************************
"""

    def run(self):
        try:
            build_ext.run(self)
        except Exception:
            e = sys.exc_info()[1]
            sys.stdout.write('%s\n' % str(e))
            warnings.warn(
                self.warning_message +
                "Extension modules" +
                "There was an issue with your platform configuration - see above.")

    def build_extension(self, ext):
        name = ext.name
        try:
            build_ext.build_extension(self, ext)
        except Exception:
            e = sys.exc_info()[1]
            sys.stdout.write('%s\n' % str(e))
            warnings.warn(
                self.warning_message +
                "The %s extension module" % (name,) +
                "The output above this warning shows how the compilation failed.")

    # the following is needed to be able to add numpy's include dirs... without
    # importing numpy directly in this script, before it's actually installed!
    # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # https://docs.python.org/2/library/__builtin__.html#module-__builtin__
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False

        import numpy
        self.include_dirs.append(numpy.get_include())

ez_setup.use_setuptools()

model_dir = os.path.join(os.path.dirname(__file__), 'fse')
cmdclass = {'build_ext': custom_build_ext}

WHEELHOUSE_UPLOADER_COMMANDS = {'fetch_artifacts', 'upload_all'}
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader.cmd
    cmdclass.update(vars(wheelhouse_uploader.cmd))


setup(
    name='FastSentenceEmbeddings',
    version='0.1',
    description='Fast Sentence Embeddings for Gensim',

    ext_modules=[
        Extension('fse.sif_variants_cy',
            sources=['./fse/sif_variants_cy.c'],
            include_dirs=[model_dir]),
        Extension('fse.sif_inner',
            sources=['./fse/sif_inner.c'],
            include_dirs=[model_dir]),
    ],
    cmdclass=cmdclass,
    packages=find_packages(),

    author=u'Oliver Borchers',
    author_email='borchers@bwl.uni-mannheim.de',

    install_requires=[
        'numpy >= 1.11.3',
        'scipy >= 0.18.1',
        'six >= 1.5.0',
        'smart_open >= 1.2.1',
    ],
)
