import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import datetime


tm = ""
dt = QDate.currentDate()
tmfin = ""
dtfin = QDate.currentDate()

class EventDetailsPicker(QWidget):
   def __init__(self):
      super(EventDetailsPicker, self).__init__()
      self.initUI()
        
   def initUI(self):
      global ruta
      cal = QCalendarWidget(self)
      cal.setGridVisible(True)
      cal.move(20, 20)
      cal.clicked[QDate].connect(self.showDate)
        
      self.date = cal.selectedDate()
      self.lbl = QLabel(self)
      self.lbl.setText(self.date.toString())
      self.lbl.move(100, 235)
      self.lbl.resize(self.lbl.sizeHint())

      self.timeq = QTimeEdit(self)
      self.timeq.move(100, 200)
      self.timeq.mousePressEvent=self.showTime

      self.lblt = QLabel(self)
      self.time = self.timeq.time()
      self.lblt.setText(self.time.toString())
      self.lblt.move(100, 250)

      cal2 = QCalendarWidget(self)
      cal2.setGridVisible(True)
      cal2.move(350+20, 20)
      cal2.clicked[QDate].connect(self.showDateFin)
    
      self.date2 = cal2.selectedDate()
      self.lbl2 = QLabel(self)
      self.lbl2.setText(self.date2.toString())
      self.lbl2.move(350+100, 235)
      self.lbl2.resize(self.lbl2.sizeHint())


      self.timeq2 = QTimeEdit(self)
      self.timeq2.move(350+100, 200)
      self.timeq2.mousePressEvent=self.showTimeFin

      self.lblt2 = QLabel(self)
      self.time2 = self.timeq2.time()
      self.lblt2.setText(self.time2.toString())
      self.lblt2.move(350+100, 250)

      qbtn = QPushButton('Guardar', self)
      qbtn.clicked.connect(QCoreApplication.instance().quit)
      qbtn.resize(qbtn.sizeHint())
      qbtn.move(300, 250)

      qbtnSet = QPushButton('Establecer hora', self)
      qbtnSet.clicked.connect(self.setTime)
      qbtnSet.resize(qbtnSet.sizeHint())
      qbtnSet.move(300, 210)
      

      self.setGeometry(100,100,700,300)
      self.setWindowTitle('Escoger tiempos evento')
      self.show()
        
   def showDate(self, d):
      global dt
      self.lbl.setText(d.toString())
      self.lbl.resize(self.lbl.sizeHint())
      dt = d

   def showTime(self, event):
      global tm
      self.lblt.setText(self.timeq.time().toString())
      tm = self.timeq.time()

   def showDateFin(self, d):
      global dtfin
      self.lbl2.setText(d.toString())
      self.lbl.resize(self.lbl.sizeHint())
      dtfin = d

   def showTimeFin(self, event):
      global tmfin
      self.lblt2.setText(self.timeq2.time().toString())
      tmfin = self.timeq2.time()

   def setTime(self):
      global tm
      global tmfin
      self.lblt2.setText(self.timeq2.time().toString())
      self.lblt.setText(self.timeq.time().toString())
      tm = self.timeq.time()
      tmfin = self.timeq2.time()


        
def main():
   app = QApplication(sys.argv)
   ex = EventDetailsPicker()
   app.exec_()
   if dt == "" or dtfin == "" or tm == "" or tmfin == "":
      return False, False ,  False
   else:
      initial = datetime.datetime.combine(dt.toPyDate(), tm.toPyTime())
      final = datetime.datetime.combine(dtfin.toPyDate(), tmfin.toPyTime())
      return initial, final, True 

    
if __name__ == '__main__':
   initial, final = main()
   print(initial, final)
