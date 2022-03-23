import cv2

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from window import Interface
from action.file_io_action import FileIOAction


class Action(FileIOAction, Interface):
    def __init__(self):
        super(Action, self).__init__()

    def close_window(self):
        """Close application"""
        self.window.close()
