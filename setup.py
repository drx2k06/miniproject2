from setuptools import setup, find_packages

setup(
    name="crediedge",
    version="2.0.0",
    description="CrediEdge AI: Autonomous Credit Risk Underwriting & Loan Eligibility Decisioning Engine",
    author="Daniel Richard Xavier",
    author_email="drx2023l@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "crediedge=src.cli:main",
        ],
    },
)
