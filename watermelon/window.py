import  os
import  sys

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *

from    interpreter.ipython_console     import  QIPythonWidget


class Structure(object):
    ''' Application window class
    '''
    def __init__(self):
        # Declare application 
        self.application = QApplication(sys.argv)
        
        # Declare main window
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("WATERMELON")
        
        # Main interface layout
        self.main_frame = QGridLayout()
        
        # Build widgets
        self.build()
        
    def build(self):
        ''' Build application
        '''
        # Build menu bar
        self.build_menu_bar()

        # Build application interface
        self.build_interface()

    def build_menu_bar(self):
        ''' Build ingredients of menu bar
        '''
        # File menu; open, close, save data
        self.setup_file_io_menu()

        # Toolkit menu
        self.setup_toolkit_menu()

    def build_interface(self):
        ''' Set up interface
        '''
        # Inteface ingradients
        interface_widget = QWidget()
        interface_frame = QGridLayout()

        # Build widgets
        self.setup_pyconsole_widget()
        #self.setup_chart_graph_widget()

        # Build frame
        interface_frame.addLayout(self.pyconsole_layout, 0, 0, 1, 1)

        # Add layouts to window
        interface_widget.setLayout(interface_frame)
        self.main_window.setCentralWidget(interface_widget)


class Interface(Structure):
    ''' Interface class
    '''
    def __init__(self):
        super(Interface, self).__init__()

    def setup_file_io_menu(self):
        ''' Set up file menu
        '''
        # Build file menu
        file_io_menu = self.main_window.menuBar().addMenu("&File")

        # Open button
        open_button = QAction(
            QIcon("exit.png"), "Open file", self.main_window
        )
        open_button.setShortcut("Ctrl+O")
        open_button.setStatusTip("Open data file")
        open_button.triggered.connect(self.open_file)

        # Save button
        save_button = QAction(
            QIcon("exit.png"), "Save file", self.main_window
        )
        save_button.setShortcut("Ctrl+S")
        save_button.setStatusTip("Save result")
        save_button.triggered.connect(self.save_file)

        # Close button
        close_button = QAction(
            QIcon("exit.png"), "Exit", self.main_window
        )
        close_button.setShortcut("Ctrl+Q")
        close_button.setStatusTip("Exit application")
        close_button.triggered.connect(self.close_window)

        # Add buttons
        file_io_menu.addAction(open_button)
        file_io_menu.addAction(save_button)
        file_io_menu.addAction(close_button)

    def setup_toolkit_menu(self):
        ''' Set up toolkit menu
        '''
        # Build toolkit menu
        toolkit_menu = self.main_window.menuBar().addMenu("&Tool")    

    def setup_pyconsole_widget(self):
        ''' Setup python console layout
        '''
        # Set python console layout
        self.pyconsole_layout = QHBoxLayout()
         
        frame_layout = QVBoxLayout()
        frame = QFrame()
        frame.setFrameShape(QFrame.Panel)

        # IPython console
        self.ipyconsole = QIPythonWidget(
            customBanner="Welcome to the WATERMELON"
        )
        self.ipyconsole.setMinimumHeight(128)

        # Set initial import
        self.ipyconsole.execute_command(
            "from interpreter.user_defined_utils import *"
        )
        self.ipyconsole.execute_command(
            "%matplotlib inline"
        )

        # Add widgets to layout
        frame_layout.addWidget(self.ipyconsole)
        frame.setLayout(frame_layout)
        self.pyconsole_layout.addWidget(frame)

        # e.g.) push variables to kernel
        #   variables = {"var1": var1, "var2": var2, ...}
        #   self.ipyconsole.push_variable(variables)
