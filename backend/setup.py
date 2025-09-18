from setuptools import setup, find_packages

setup(
    name="minigpt",
    version="0.1.0",
    description="A minimalist GPT implementation for learning and experimentation",
    author="Benjamin Hu",
    author_email="benjamin@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.21.0",
        "tokenizers>=0.13.0",
        "datasets>=2.0.0",
        "tqdm>=4.64.0",
        "wandb>=0.13.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "minigpt-train=minigpt.train:main",
            "minigpt-chat=minigpt.chat:main",
        ],
    },
)