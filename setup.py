"""
Setup script for the LLM Cost Recommendation package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')

setup(
    name="llm-cost-recommendation",
    version="1.0.0",
    author="AWS Cost Optimization Team",
    author_email="nggocnn@example.com",
    description="A multi-agent system for AWS cost optimization using Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nggocnn/llm-cost-recommendation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-cost-recommendation=llm_cost_recommendation.console:main_sync",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_cost_recommendation": ["py.typed"],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
    },
    keywords="aws cost optimization llm ai machine-learning cloud finops",
    project_urls={
        "Bug Reports": "https://github.com/nggocnn/llm-cost-recommendation/issues",
        "Source": "https://github.com/nggocnn/llm-cost-recommendation",
        "Documentation": "https://github.com/nggocnn/llm-cost-recommendation/blob/main/README.md",
    },
)
