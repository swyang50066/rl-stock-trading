import  sys

from    window          import  Interface, Structure
from    action.action   import  Action


class Watermelon(Action, Interface, Structure):
    def __init__(self):
        super(Watermelon, self).__init__()

    def execute(self):
        ''' Execute WATERMELON
        '''
        # Apply style sheet
        # UI design is to be updated later.

        # Run application
        self.main_window.show()
        sys.exit(self.application.exec_())


if __name__ == "__main__":
    watermelon = Watermelon()
    watermelon.execute()
