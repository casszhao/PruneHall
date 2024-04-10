from setuptools import setup, find_packages

setup(
    name="pruning-study",
    version="0.1.0",
    author="Gchrysostomou, mlsw, casszhao",
    author_email="george1bodom@gmail.com, mwilliams15@sheffield.ac.uk, zhixue.zhao@sheffield.ac.uk",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["pruning_study"]),
    include_package_data=True,
    python_requires=">=3.10, <3.13",
    install_requires=[
        "loguru>=0.7.2,<1.0.0",
        "torch>=2.2.2,<3.0.0",
        "nltk>=3.8.1,<4.0.0",
        "sentencepiece>=0.2.0,<1.0.0",
        "protobuf>=5.26.1,<6.0.0",
        "evaluate>=0.4.1,<1.0.0",
        "datasets>=2.18.0,<3.0.0",
        "huggingface-hub>=0.22.2,<1.0.0",
        "accelerate>=0.29.1,<1.0.0",
        "bert-score>=0.3.13,<1.0.0",
        "absl-py>=2.1.0,<3.0.0",
        "rouge-score>=0.1.2,<1.0.0",
        "pydantic>=2.6.4,<3.0.0",
        "scikit-learn>=1.4.2,<1.5",
        "openpyxl>=3.1"
    ],
    extras_require={
        "dev": [
            "ruff>=0.1.5,<1.0.0",
            "pytest>=7.4.3,<8.0.0",
            "black>=23.11.0,<24.0.0",
            "mypy>=1.7.0,<2.0.0",
            "pre-commit>=3.7.0,<4.0.0",
            "poetry-plugin-export>=1.7.1,<2.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)