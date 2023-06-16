import os

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)