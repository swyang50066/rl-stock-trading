from qtconsole.rich_ipython_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager


class QIPythonWidget(RichJupyterWidget):
    """Convenience class for a live IPython console widget.

    We can replace the standard banner using the custom_banner argument
    """

    def __init__(self, custom_banner=None, *args, **kwargs):
        super(QIPythonWidget, self).__init__(*args, **kwargs)

        # Set customized banner
        if custom_banner != None:
            self.banner = custom_banner

        # Define kernel and launch kernel
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = "qt"

        # Define user and start channel
        self.kernel_client = kernel_client = self.kernel_manager.client()
        kernel_client.start_channels()

        # Connect stop signal
        self.exit_requested.connect(self.stop)

    def stop(self):
        """Stop console"""
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()

    def push_variable(self, variables):
        """Given a dictionary containing name / value pairs,
        push those variables to the IPython console widget.
        """
        self.kernel_manager.kernel.shell.push(variables)

    def clear_terminal(self):
        """Clears the terminal"""
        self._control.clear()

    def print_text(self, text):
        """Prints text to the console"""
        self._append_plain_text(text)

    def execute_command(self, command):
        """Execute a command in the frame of the console widget"""
        self._execute(command, False)
