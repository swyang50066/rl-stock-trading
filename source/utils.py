import os


def make_directory(folder_path):
    """Make directory"""
    return os.mkdir(folder_path) if not os.path.isdir(folder_path) else None


def join_path_item(*args):
    """Join path items according to os file system"""
    return os.path.join(*args)
