from pathlib import Path
from setuptools import setup, find_packages, Command
import subprocess
import os

__version__ = 1.0

class InstallSubmodules(Command):
    """A custom command to install and build submodules."""
    description = 'Install and build submodules'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Define the submodules and their setup commands
        submodules = {
            'hloc': {
                'path': 'code/0-calibration/hloc',
                'commands': [
                    ['pip', 'install', '-e', '.'],
                ]
            },
            'multiviewcalib': {
                'path': 'code/0-calibration/multiview_calib',
                'commands': [
                    ['pip', 'install', '-e', '.'],
                ]
            },
            
        }

        # Run the setup commands for each submodule
        for name, details in submodules.items():
            submodule_path = details['path']
            for command in details['commands']:
                self.run_command_in_submodule(submodule_path, command)

    def run_command_in_submodule(self, submodule_path, command):
        """Run a command inside a submodule's directory."""
        original_dir = os.getcwd()
        os.chdir(submodule_path)
        try:
            self.announce(f'Running command: {" ".join(command)} in {submodule_path}', level=3)
            subprocess.check_call(command)
        finally:
            os.chdir(original_dir)

# Read the long description from the README file
root = Path(__file__).parent
with open(root / 'README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the dependencies from the requirements.txt file
with open(root / 'requirements.txt', 'r', encoding='utf-8') as f:
    dependencies = f.read().splitlines()

setup(
    name='MARMOT',
    version=__version__,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=dependencies,
    author='Martin Engilbere, Wilke Grosche',
    description='Multiview Tracking Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cvlab-epfl/MARMOT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'install_submodules': InstallSubmodules,
    },
)
