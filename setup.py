"""Setup script for SGNLite package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sgnlite",
    version="0.1.0",
    author="Brian",
    author_email="",
    description="A lightweight transformer for skeleton-based tennis swing detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sgnlite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "full": [
            "ultralytics>=8.0.0",
            "ffmpeg-python>=0.2.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "scikit-learn>=0.24.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sgnlite-train=scripts.train:main",
            "sgnlite-infer=sgnlite.inference:main",
        ],
    },
)
