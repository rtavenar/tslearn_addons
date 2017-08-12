`tslearn_addons` is a Python package that provides additional methods to be used with [`tslearn`](https://github.com/rtavenar/tslearn).

# Dependencies

```
Cython
numpy
scipy
scikit-learn
tslearn
```

# Installation

## Using latest github-hosted version

If you want to get `tslearn_addons`'s latest version, you can `git clone` the repository hosted at github:
```bash
git clone https://github.com/rtavenar/tslearn_addons.git
```

Then, you should run the following command for Cython code to compile:
```bash
python setup.py build_ext --inplace
```

Also, for the whole package to run properly, its base directory should be appended to your Python path.
