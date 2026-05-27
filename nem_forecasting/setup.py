from setuptools import setup, find_packages

setup(
    name="nem-forecasting",
    version="0.1.0",
    description="End-to-end ML pipeline for Australian NEM electricity price forecasting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "mysql-connector-python>=8.3.0",
        "sqlalchemy>=2.0.0",
        "streamlit>=1.35.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nem-pipeline=run_pipeline:execute_system_pipeline",
        ],
    },
)