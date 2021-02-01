# CSGO Grenade classifier

Project created for 'esportsLABgg Counter-Strike Data Challenge' Data Analysis challenge
 
# Topic

The aim is to classify a grenade throw as either valid or invalid, using a model trained on two 
sets of grenade throws (from two different maps), and then to append the prediction result to the test data.

# Install dependencies

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

# Create jupyter kernel

To use external modules in the jupyter notebook, a kernel with project's dependencies needs to be created first.

Do it using the following command

```
python -m ipykernel install --user --name [myenv] 
```

where [myenv] is how you want to name the kernel.

# Use jupyter

To view, edit and run project's jupyter notebooks, in the command line (while in project's root) run:

``` 
jupyter notebook
```

A notebook browser should open in your web browser.

Inside the notebook, remember to select the kernel you created with this project's environment.

### Just browse the notebooks

The notebooks have been exported as html to /notebooks, if you just want to read their contents
without interacting 

# Running the classifier

To run the classifier, make sure you have the pipenv shell enabled. Then, move to src:

```
cd src
```

This is where the classify.py script is located.

To use the script:

```
python classify.py [filepath]
```

where [filepath] is the path to the file that you want to add predictions to.
For example, if you want to predict for a file test.csv, which is in the same directory as classify.py:

``` 
python classify.py test.csv
```

test.csv will now have a new column called RESULT, which can have values TRUE or FALSE, 
depending on whether the grenade was predicted to be valid or not.