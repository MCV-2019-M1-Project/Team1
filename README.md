# Team 1

TODO

# Getting started

On your local machine run the following command
```
pip3 install virtualenv
```

To be able to run the program you've to do the following
## Windows
```
virtualenv env -p python
env\Scripts\activate.bat
pip3 install -r requirements.txt
python src/museum.py
env\Scripts\deactivate.bat
```

## Linux
```
virtualenv env -p python3
source env/bin/activate
pip3 install -r requirements.txt
python3 src/museum.py
deactivate
```

