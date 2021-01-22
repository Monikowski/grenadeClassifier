#CSGO Grenade classifier

Project created for 'esportsLABgg Counter-Strike Data Challenge' Data Analysis challenge
 
#Topic

The aim is to classify a grenade throw as either valid or invalid, using a model trained on two 
sets of grenade throws (from two different maps).

#Install dependencies

Python 3.8 64bit

To install pipenv (packaging tool), run

```
pip install pipenv
```

Activate virtual environment

```
pipenv shell
```

Then, to install dependencies, run

```
pipenv install
```

#Create jupyter kernel

To use external modules in the jupyter notebook, a kernel with project's dependencies needs to be created first.

Do it using the following command

```
python -m ipykernel install --user --name [myenv] 
```

where [myenv] is how you want to name the kernel.