from setuptools import setup, find_packages

def load_reqs(fname):
    """Load a requirements file, ignore blank lines and comments."""
    with open(fname) as f:
        reqs = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            reqs.append(line)
        return reqs

# core dependencies
install_requires = load_reqs('requirements.txt')

# experiments dependencies
experiments_requires = load_reqs('requirements_experiments.txt')

# test dependencies
test_requires = ["pytest", "black", "isort"]

setup(
    name='kqe',
    version='0.1.0',
    packages=find_packages(exclude=['tests', 'experiments']),
    python_requires='>=3.8',
    install_requires=install_requires,           # ← REQUIRED
    extras_require={                             # ← OPTIONAL extras
        "dev": test_requires,
        'experiments': experiments_requires + test_requires,
    },
    license='Apache License 2.0',
    long_description=open('README.md').read(),
)
