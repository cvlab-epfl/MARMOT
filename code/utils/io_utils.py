import os
import json
import mimetypes
import shutil
import re
import numpy as np
mimetypes.init()

def is_media_file(fileName: str) -> bool:
    """
        Returns True if fileName is a media file
        (i.e. a video or image file), False otherwise.

        Arguments:
        ----------
        fileName(str): name of file
    """
    mimestart = mimetypes.guess_type(fileName)[0]

    if mimestart != None:
        mimestart = mimestart.split('/')[0]

        if mimestart in ['video', 'image']:
            return True
    
    return False


def load_json(filename: str) -> dict:

    """
        Attempts to read json file 'filename'.
        Throws an exception if unable to do so.

        Arguments:
        ----------
        filename(str): path to json file
    """

    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))


def write_json(filename: str, data: dict) -> None:

    """
        Attempts to write json file 'filename'.
        Throws an exception if unable to do so.

        Arguments:
        ----------
        filename(str): path to json file
        data(dict): data to be written to json file
    """

    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))


def rgb2gray(image):
    dtype = image.dtype
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return gray.astype(dtype)

def rmdir(directory):
    directory = os.path.abspath(directory)
    if os.path.exists(directory): 
        shutil.rmtree(directory)  

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files
     
def find_images(file_or_folder, hint=None):  
    filenames = find_files(file_or_folder, hint)
    filename_images = []
    for filename in filenames:
        _, extension = os.path.splitext(filename)
        if extension.lower() in [".jpg",".jpeg",".bmp",".tiff",".png",".gif"]:
            filename_images.append(filename)                 
    return filename_images  
