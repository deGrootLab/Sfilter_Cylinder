from setuptools import setup, find_packages

setup(
    name='Sfilter',
    version='0.3',
    description='This is a tool for basic analysis in Potassium Channel MD simulation.',
    author='Chenggong Hui',
    author_email='chenggong.hui@mpinat.mpg.de',
    packages=find_packages(),
    scripts=['script/count_cylinder.py',
             'script/match_xtck.py'
             ],
    install_requires=["MDAnalysis", "numpy", "pandas", "scipy"
        # List your package's dependencies here
    ],
)
