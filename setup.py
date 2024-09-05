# Installation script
from setuptools import setup, find_packages

# Read the contents of the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rlhf_summarization",  # Project name
    version="0.1.0",  # Initial version
    author="Arefeh Yavary",  # Author name
    author_email="your_email@example.com",  # Author email
    description="Reinforcement Learning from Human Feedback (RLHF) for Text Summarization",  # Short project description
    long_description=long_description,  # Long description from the README file
    long_description_content_type="text/markdown",  # README file format
    url="https://github.com/yourusername/rlhf_summarization",  # Project URL (GitHub, for example)
    packages=find_packages(),  # Automatically find packages in the current directory
    classifiers=[  # Optional project classifiers for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum required Python version
    install_requires=[  # List of dependencies (from requirements.txt)
        "transformers==4.16.2",
        "datasets==2.11.0",
        "torch==1.13.1",
        "tokenizers==0.13.2",
        "rouge-score==0.1.2",
        "evaluate==0.4.0",
        "einops==0.6.0",
        "numpy==1.23.1",
        "scipy==1.10.0",
        "pandas==1.4.3",
        "tqdm==4.64.0",
        "scikit-learn==1.1.1",
        "matplotlib==3.5.2",
    ],
    entry_points={  # Optional entry points for CLI scripts
        "console_scripts": [
            "train-rlhf=train:main",  # Create a CLI command 'train-rlhf' to run the train script
            "evaluate-rlhf=evaluate:main",  # Create a CLI command 'evaluate-rlhf' to run the evaluate script
        ],
    },
)
