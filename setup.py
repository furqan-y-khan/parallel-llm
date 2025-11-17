"""
Parallel-LLM: Ultra-Fast Parallel Training and Inference for Language Models
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="parallel-llm",
    version="0.1.0",
    author="Parallel-LLM Team",
    author_email="contact@parallel-llm.ai",
    description="Ultra-fast parallel training and inference for language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/furqan-y-khan/parallel-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0", "black>=23.0", "flake8>=6.0", "mypy>=1.0"],
        "docs": ["sphinx>=5.0", "sphinx-rtd-theme>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "parallel-llm-train=parallel_llm.cli:train_cli",
            "parallel-llm-infer=parallel_llm.cli:infer_cli",
        ],
    },
)
