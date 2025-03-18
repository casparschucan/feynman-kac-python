#! /bin/bash

# This script is used to set-up the SDK environment for the user.

# enter the virtual envirenment
source venv/bin/activate

# install the required packages
pip install -r requirements.txt

# enter development dir
cd src 

# start jupyter notebook
jupyter notebook experiments.ipynb
