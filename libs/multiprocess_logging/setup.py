from pathlib import Path

from setuptools import setup

libs_dir = Path(__file__).resolve().parents[1]
setup(install_requires=[f'lodis @ file://{str(libs_dir / "lodis")}'])
