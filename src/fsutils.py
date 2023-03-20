import os
import sys


def resource_file_path(filename):
    """ Search for filename in the list of directories specified in the
        PYTHONPATH environment variable.
        Taken from https://stackoverflow.com/questions/45806838/can-i-locate-resource-file-in-pythonpath
    """
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath is None:
        print(f"Specify the PYTHONPATH env variable with the directories you want the file {filename} to be searched in.")
        sys.exit(1)
    directories = pythonpath.split(os.pathsep)
    for d in directories:
        filepath = os.path.join(d, filename)
        if os.path.exists(filepath):
            return filepath

    print(f"File not found: {filename}")
    print(f"Tried the following directories:")
    print(directories)
    sys.exit(1)


def remove_all_with_extension(dir_path, extension):
    for file in os.listdir(dir_path):
        if file.endswith(extension):
            os.remove(os.path.join(dir_path, file))
