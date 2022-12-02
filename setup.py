from setuptools import setup

setup(
    name='kpi',
    version='0.1.0',
    description='Package for computing KPIs',
    install_requires=["numpy", "pandas", "scikit-learn"],
    packages=['kpi'],
    extras_require = {'docs': ['sphinx', 'sphinx_rtd_theme']},
)
