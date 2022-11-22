# DataViz 

Data Viz is a simple data visualisation tool which can be used to quickly glance through data provided via CSV files. It can be used to plot different types of graphs between pairs of parameters in the dataset. It can also be used to make a simple model fit Linear Regression.

Steps to run the application

1) Setup a Virtual Environment

    i) "pip install virtualenv" to install the virtual environment library
    ii) "virtualenv venv" to create a virtual environment
    ii) "./venv/Scripts/Activate.ps1" to activate the environment

2) One the virutal environment is activated, use " pip install -r requirements.txt " command to install all the dependencies in the environment

3) "cd dataviz" to change the directory to dataviz folder

4) "python manage.py runserver" to start the application

5) Go to "http://127.0.0.1:8000/" to access the application from any web browser

The detailed documentation about the functions and their implementation is included in the docstrings of the functions

Find the detailed document directory structure below : 

DataViz
├─ .idea
│  ├─ misc.xml
│  ├─ modules.xml
│  ├─ vcs.xml
│  └─ workspace.xml
├─ DataViz.iml
├─ README.md
├─ dataviz
│  ├─ dataviz
│  │  ├─ __init__.py
│  │  ├─ __pycache__
│  │  ├─ asgi.py
│  │  ├─ graphs.py
│  │  ├─ settings.py
│  │  ├─ urls.py
│  │  ├─ views.py
│  │  └─ wsgi.py
│  ├─ db.sqlite3
│  ├─ manage.py
│  ├─ static
│  │  ├─ bgwhite.webp
│  │  ├─ dataviz.gif
│  │  ├─ linear_regression.svg
│  │  ├─ logistic_regression.svg
│  │  ├─ sample_datasets
│  │  │  ├─ Geyser MTTF Dataset.csv
│  │  │  ├─ Height Weight Dataset.csv
│  │  │  ├─ Nile Flood Dataset.csv
│  │  │  └─ Student BMI Dataset.csv
│  │  └─ sample_datasets_images
│  │     ├─ Geyser MTTF Dataset.jpeg
│  │     ├─ Height Weight Dataset.jpeg
│  │     ├─ Nile Flood Dataset.jpeg
│  │     └─ Student BMI Dataset.jpeg
│  └─ templates
│     ├─ index.html
│     ├─ linear_regression.html
│     ├─ logistic_regression.html
│     └─ show_data.html
├─ include
└─ requirements.txt