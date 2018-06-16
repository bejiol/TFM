import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ScrollMessageBox(QMessageBox):
   def __init__(self, number ,l, *args, **kwargs):
      QMessageBox.__init__(self, *args, **kwargs)
      scroll = QScrollArea(self)
      scroll.setWidgetResizable(True)
      self.content = QWidget()
      scroll.setWidget(self.content)
      lay = QVBoxLayout(self.content)
      for item in l:
         lay.addWidget(QLabel(item, self))
      self.layout().addWidget(scroll, 0, 0, 1, self.layout().columnCount())
      self.setStyleSheet("QScrollArea{min-width:700 px; min-height: 500px}")
      self.setWindowTitle("Tema " + str(number))


if __name__ == "__main__":
   app = QApplication(sys.argv)
   gui = ScrollMessageBox(lst, None)
   gui.exec_()
   sys.exit(app.exec_())