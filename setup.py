from setuptools import setup

setup(
    name="copula-shirley",
    version="1.0.0",
    packages=["copula_shirley"],
    install_requires=[
        "category-encoders==2.5.1.post0",
        "diffprivlib==0.6.0",
        "numpy==1.23.4",
        "pandas==1.5.1",
        "pyvinecopulib==0.6.2",
        "scikit-learn==1.1.3",
        "scipy==1.9.3",
        "xgboost==1.7.0",
    ],
)
