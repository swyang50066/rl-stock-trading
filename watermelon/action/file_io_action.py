import cv2
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from method.file_io import FileIO


class FileIOAction(object):
    def __init__(self):
        super().__init__()

    def open_file(self):
        """Open input file"""
        # Read file path
        input_file_paths = QFileIODialog.getOpenFileIOName()

        # No path given
        if not input_file_paths:
            return 0

    def save_file(self):
        """Save output file"""
        # Read output path
        output_file_paths = QFileIODialog.getExistingDirectory()
