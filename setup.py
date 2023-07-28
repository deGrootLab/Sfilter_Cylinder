from setuptools import setup, find_packages
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='Sfilter',
    version=get_version("Sfilter/__init__.py"),
    description='This is a tool for basic analysis in Potassium Channel MD simulation.',
    author='Chenggong Hui',
    author_email='chenggong.hui@mpinat.mpg.de',
    packages=find_packages(),
    scripts=['script/count_cylinder.py', 'script/match_xtck.py'],
    install_requires=["MDAnalysis", "numpy", "pandas", "scipy", "networkx", "matplotlib", "pyemma"],
)
