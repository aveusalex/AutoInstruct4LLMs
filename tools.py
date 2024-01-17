import os


def check_path_exists(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)
