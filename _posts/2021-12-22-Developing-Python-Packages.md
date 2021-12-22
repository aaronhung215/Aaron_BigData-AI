---
layout: xxxx
title: Developing Python Packages
date: 2021-12-22
tags: datacamp, python, test
comments: true
---
# Developing Python Packages

## Lesson 1
* reuse
* key function up to date
* sharing

* Terms
	* script : python file
	* package : a directory full of python code to be imported, such as numpy
	* subpackage,  such as numpy.random
	* Module : a python file inside a package which stores the package code
	* library : a package or collection of package

* Directory tree of package 
	* simplemodule.py
	* __init__.py :

![](https://i.imgur.com/oTAF70Y.png)


* from scrip to package

![](https://i.imgur.com/4XkbFhs.png)


### Documentation
* Function
* Class
* Class method

![](https://i.imgur.com/qeomkv9.png)



* docstring templates

![](https://i.imgur.com/BEsFFBQ.png)


` pyment -w -o google textanalysis.py `

![](https://i.imgur.com/zWOB0mH.png)



```python
def inches_to_feet(x, reverse=False):
    """Convert lengths between inches and feet.

    Parameters
    ----------
    x : numpy.ndarray
        Lengths in feet.
    reverse : bool, optional
        If true this function converts from feet to inches 
        instead of the default behavior of inches to feet. 
        (Default value = False)

    Returns
    -------
    numpy.ndarray
    """
    if reverse:
        return x * INCHES_PER_FOOT
    else:
        return x / INCHES_PER_FOOT

```


![](https://i.imgur.com/hSuj8uS.png)

![](https://i.imgur.com/Jp3gBhI.png)

![](https://i.imgur.com/rD0JvfW.png)


### Structuring imports

```python
import mysklearn.preprocessing.normalize

from mysklearn import preprocessing
```

![](https://i.imgur.com/oJIPmpD.png)



```python

"""User-facing functions."""
from impyrial.length.core import (
    UNITS,
    inches_to_feet,
    inches_to_yards
)

```


## Lesson 2 : install your own package

![](https://i.imgur.com/I69se2m.png)


* Package directory structure
![](https://i.imgur.com/pHpHqO5.png)


* setup.py

![](https://i.imgur.com/4u0EFXw.png)


* find_package

[image:3ECF5F03-2C74-4140-9642-8F67E1DFE5B9-23475-00000614EA62826E/截圖 2021-12-20 下午11.15.26.png]

```python

pip install -e .
```

### dealing with dependencies

![](https://i.imgur.com/ZrdkkRx.png)


![](https://i.imgur.com/oafu648.png)


```
pip freeze > requirements.txt

pip install - r requirements.txt
```

![](https://i.imgur.com/sTcPC6p.png)


### Including licences and writing READMEs
* license
	* to give others permission to use your code

* README
	* Title
	* Description
	* Installation
	* Usage examples
	* Contributing
	* License

![](https://i.imgur.com/KMD89uU.png)


* MANIFEST.in
	* List all the extra files to include in your package distribution
	* include README.md
	* include LICENSE

### Publishing your package
* PyPI
* Distributions
	* source distribution : a distribution package which is mostly your source code
	* wheel distribution : a distribution package which has been processed to make it faster to install

```
python setup.py sdist bdist_wheel

```


![](https://i.imgur.com/mILmypC.png)


* upload
```python
twine upload dist/*

twine upload -r testpypi dist/*
```

## Lesson 3 : Testing your package
* Writing tests

![](https://i.imgur.com/jiPxE8c.png)


* Organizing tests inside your package

![](https://i.imgur.com/OSHL0IY.png)


``` 
pytest
# 會執行所有test開頭的module
```


![](https://i.imgur.com/BPigsJn.png)


### Testing your package with different environments
* Testing multiple versions of Python
	* tox
		* tox.ini
			![](https://i.imgur.com/vALbdCL.png)

	
			![](https://i.imgur.com/bTIJBB0.png)

			
            ![](https://i.imgur.com/0PpE2eL.png)


### keeping your package stylish
* standard python style is described in PEP8
* a style guide dictates how code should be laid out
	* flake8
` flake8 features.py `

![](https://i.imgur.com/73NKpmS.png)

	* breaking the rule on purpose

![](https://i.imgur.com/AfsVTH7.png)

	* find/ignore specific rule
![](https://i.imgur.com/pYW9znU.png)


* setup.cfg

![](https://i.imgur.com/QULqZxU.png)


```
[flake8]

# Ignore F401 violations in the main __init__.py file
# per-file-ignores =
#     impyrial/__init__.py : F401
        
# Ignore all violations in the tests directoory
# exclude = tests/*

```


## Lesson 4: Rapid Package Development
* Faster package development with templates
* cookiecutter
` cookiecutter https://github.com/audreyr/cookiecutter-pypackage `

![](https://i.imgur.com/FNpXTqo.png)


### version numbers and history
* CONTRIBUTING.md
* HISTORY.md
	* e.g. NumPy release notes
> # History
> ## 0.3.0
> ### Changed
> - ….
> ### Deprecated
> -  …
> ## 0.2.1
> ### Fixed
> - …
> ## 0.2.0
> ### Added
> - ….	
> ### Deprecated
> - …
> - …

* Version number
	* setup.py
	* __init__.py
	* bumpversion
		* major
		* minor
		* patch

### Makefiles and classifier
* Classifiers
	* inside `setup.py` of mysklearn

![](https://i.imgur.com/3kAueRH.png)


* Makefiles
	* used to automate parts of building your package
		* make clean
		* make test
		* make dist


![](https://i.imgur.com/Jay9YWW.png)


* Advanced: 
	* UT in DS
	* Package website : ReadtheDocs and Sphinx