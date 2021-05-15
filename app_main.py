import sys

from PyQt5.QtWidgets import *


from UI_MainWindow import MainWindow

if __name__=='__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()  
    demo.showMaximized()    
    sys.exit(app.exec_())