from setuptools import setup, find_packages

setup(
    name="medical-image-lib",
    version="1.0.0",
    author="Emmanuel Sande",
    author_email="emmanuelsande14@gmail.com",
    description="A Python library for medical image classification with limited data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=1.0.0",
        "albumentations>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    keywords="medical imaging, machine learning, deep learning, transfer learning, few-shot learning",
    project_urls={
        "Bug Reports": "https://github.com/emmanuelsande/medical-image-lib/issues",
        "Source": "https://github.com/emmanuelsande/medical-image-lib",
    },
)

