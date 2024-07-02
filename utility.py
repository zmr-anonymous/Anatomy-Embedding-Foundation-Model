import sys
import os
import json
import re
from time import time, sleep
from datetime import datetime
from dataclasses import dataclass
import importlib
import pkgutil

# shortening
join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir

def subdirs(folder: str, join=True, prefix=None, suffix=None, sort=True):
    ''' Get lists of subdirs for a target folder.

    Args:
        folder: (str), Path of target folder.
        join: (bool), Return full path of subfolders (True) or folder names (False). The default is True.
        prefix: (str), Specify prefix.
        suffix: (str), Specify suffix.
        sort: (bool), Whether to sort the results. The default is True.

    Returns:
        (list) subdirs list.
    '''
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

subfolders = subdirs

def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    ''' Get lists of sub-files for a target folder.

    Args:
        folder: (str), Path of target folder.
        join: (bool), Return full path of sub-files (True) or file names (False). The default is True.
        prefix: (str), Specify prefix.
        suffix: (str), Specify suffix.
        sort: (bool), Whether to sort the results. The default is True.

    Returns:
        (list) subdirs list.
    '''
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def maybe_mkdir_p(directory):
    ''' If a path does not exist, create it level by level.

    Args:
        directory: (str), Target path.

    Raises:
        FileExistsError: This can sometimes happen when two jobs try to create the same directory at the same time.
    '''
    directory = os.path.abspath(directory)
    # splits = directory.split("/")[1:]
    splits = re.split(r'[/\\]', directory)[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

def load_json(file):
    '''

    Args:
        file:

    Returns:

    '''
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def save_json(obj, file, indent=4, sort_keys=True):
    '''

    Args:
        obj:
        file:
        indent:
        sort_keys:

    Returns:

    '''
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

@dataclass
class globalVal():
    log_file = ''
    project_path = ''
    model_path = ''
    output_path = ''
    device = ''
    first_rank = True

def init_log_file(path:str, prefix='mrLog'):
    """ 
    init log file, call this function at the begining fo the programme. After this \
        you can call print_to_log_file to write text into the log file.

    Args:
        path: (str), folder to write log
        prefix: (str), prefix of the log file name
    Returns:
        
    """
    maybe_mkdir_p(path)
    fileName = datetime.now().strftime(prefix+'_'+'%Y_%m_%d_%H_%M_%S')
    globalVal.log_file = join(path, fileName + '.txt')

    with open(globalVal.log_file, 'w') as f:
        f.write("Starting... \n")


def print_to_log_file(*args, also_print_to_console=True, add_timestamp=True):
    """ 
    print to log file. 

    Args:
        *args: things to print
        also_print_to_console: (bool) 
        add_timestamp: (bool), add time stamp at beginning of the line
    Returns:
        
    """
    if not globalVal.first_rank:
        return
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = ("%s:" % dt_object, *args)

    assert globalVal.log_file is not None, "error: print to log file before init. (from utility.py)"
    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(globalVal.log_file, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)

def recursive_find_class(folder, class_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder[0]]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder[0]]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_class([join(folder[0], modname)], class_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr