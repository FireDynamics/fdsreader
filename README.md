# FDSReader
> Fast and easy-to-use Python reader for FDS data

[![PyPI version](https://badge.fury.io/py/fdsreader.png)](https://badge.fury.io/py/fdsreader)  


## Installation

The package is available on PyPI and can be installed using pip:  
```sh
pip install fdsreader
```
_FDS Version 6.7.5 and above are fully supported. Versions below 6.7.5 might work, but are not guaranteed to work._

## Usage example

```python
import fdsreader as fds

# Creates an instance of a simulation master-class which manages all data for a given simulation
sim = fds.Simulation("./sample_data")

# Examples of data that can be easily accessed
print(sim.meshes, sim.surfaces, sim.slices, sim.data_3d, sim.smoke_3d, sim.isosurfaces, sim.particles, sim.obstructions)
```

More advanced examples can be found in the respective data type directories inside of the examples directory.  

### Configuration
The package provides a few configuration options that can be set using the `settings` module.  
```python
fds.settings.KEY = VALUE

# Example
fds.settings.DEBUG = True
```  

|      KEY       |  VALUE  | Default | Description |
|----------------|---------|---------|-------------|
|    LAZY_LOAD   | boolean |   True  | Load all data when initially loading the simulation (False) or only when specific data is needed (True). |
| ENABLE_CACHING | boolean |   True  | Cache the loaded simulation to reduce startup times when loading the same simulation again. |
|     DEBUG      | boolean |  False  | Crash on non-critical errors with an exception (True) or output non-critical errors as warnings (False). |
| IGNORE_ERRORS  | boolean |  False  | Ignore any non-critical errors completely. |


### Data structure
![Data structure](https://raw.githubusercontent.com/FireDynamics/fdsreader/master/docs/img/data-structure.svg)

Beware that not all attributes and methods are covered in this diagram. For a complete  
documentation of all classes check the API Documentation below.  

## API Documentation
[https://firedynamics.github.io/fdsreader/](https://firedynamics.github.io/fdsreader/)

## Deployment
As the fdsreader has come a long way and the free capabilities of Travis CI have been used up, we now moved to manual CI/CD using a local docker container.  
First, the Dockerfile has to be modified to make authentication to GitHub and PyPI possible from within the container.
To do so generate these two tokens:  
PyPI: https://pypi.org/manage/account/token/  
GitHub: https://github.com/settings/tokens/new (set the repo_deployment and public_repo scopes)  
Now add these Tokens in the Dockerfile. To now deploy the fdsreader to PyPI and update the Github Pages (Documentation), run the following commands after pushing your changes to the FDSReader to GitHub (apart from the Dockerfile).
```bash
cd $REPO_ROOT_DIR
docker build . -t fdsreader-ci  # Only needed the very first time
docker run --rm fdsreader-ci
```

### Manual deployment
It is also possible to deploy to PyPI and Github pages manually using the following steps:
1. python setup.py sdist bdist_wheel
2. twine upload dist/*
3. sphinx-build -b html docs docs/build
4. cd .. && mkdir gh-pages && cd gh-pages
5. git init && git remote add origin git@github.com:FireDynamics/fdsreader.git
6. git fetch --all
7. git checkout gh-pages
8. cp -r ../docs/build/* .
9. git add . && git commit -m "..." && git push origin HEAD:gh-pages

## Meta

*  Jan Vogelsang – j.vogelsang@fz-juelich.de
*  Prof. Dr. Lukas Arnold - l.arnold@fz-juelich.de

Distributed under the LGPLv3 (GNU Lesser General Public License v3) license. See ``LICENSE`` for more information.

[https://github.com/FireDynamics/fdsreader](https://github.com/FireDynamics/fdsreader)

## Contributing

1. Fork it (<https://github.com/FireDynamics/fdsreader/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
