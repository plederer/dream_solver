## Installation

To install the *dream* package clone the repository to your local machine.
```{code-block} console
(dream) user:~$ git clone git@github.com:plederer/dream_solver.git ~/target-path
```
Run the following command in the directory of the git repository:
```{code-block} console
(dream) user:~$ python3 -m pip install -e ~/target-path/.
```
This will install the *dream* package and all its dependencies from pip including [`ngsolve`](https://ngsolve.org/).

``````{tip} 
We advice the use of a virtual environment for the installation of the *dream* package
and its dependencies.

To setup and activate a virtual environment run the following commands:
```{code-block} console
user:~$ python3 -m venv ~/env-path/dream
user:~$ source ~/env-path/dream/bin/activate
(dream) user:~$  which python3
/home/user/env-path/dream/bin/python3
```
To deactivate the virtual environment run:
```{code-block} console
(dream) user:~$ deactivate
user:~$ 
```
``````

```{note}
If the installation finds an underlying installation of `ngsolve` it will not install the package from pip, but use the existing installation.
```
```{note}
The *dream* package requires Python 3.10 and later versions.
```