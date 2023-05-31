from setuptools import setup, find_packages

setup(
    name='Sfilter',
    version='0.1',
    description='This is a tool for basic analysis in Potassium Channel MD simulation.',
    author='Chenggong Hui',
    author_email='chenggong.hui@mpinat.mpg.de',
    packages=find_packages(),
    scripts=['script/count_cylinder.py'],
    install_requires=["MDAnalysis"
        # List your package's dependencies here
    ],
)
