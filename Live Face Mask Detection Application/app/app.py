from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2
from face_mask_detector import face_mask_prediction

class VideoCapture(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)
    
    def __init__(self): 
        super().__init__()
        self.run_flag = True
    
    def run(self):
        cap = cv2.VideoCapture(0)
       
        while self.run_flag:
            ret , frame = cap.read() # BGR Format
            pred_img = face_mask_prediction(frame)
            
            if ret == True:
                self.change_pixmap_signal.emit(pred_img)

        cap.release()

    def stop(self):
        self.run_flag = False
        self.wait() 

class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(qtg.QIcon('./Images/6861692.png'))
        self.setWindowTitle('Live Face Mask Detection')
        self.setFixedSize(600,600)
        
        
        label = qtw.QLabel('<h2>Face Mask Detection Application</h2>')
        
        self.cameraButton = qtw.QPushButton('Click to Open Camera',clicked=self.cameraButtonClick,checkable=True)
        
        self.screen = qtw.QLabel()
        self.img = qtg.QPixmap(600,600)
        self.img.fill(qtg.QColor('darkGrey')) 
        self.screen.setPixmap(self.img)
        
        
        text = qtw.QLabel("This Software is a real-time application to detect people if they are wearing a mask or are without a mask through the help of Webcam.") 
        text.setWordWrap(True)
        
        #Layout
        layout = qtw.QVBoxLayout()
        layout.addSpacing(25)
        layout.addWidget(label)
        layout.addSpacing(25)
        layout.addWidget(self.cameraButton)
        layout.addWidget(self.screen)
        layout.addSpacing(5)
        layout.addWidget(text)
        
        
        self.setLayout(layout)
               
        self.show()
        
    def cameraButtonClick(self):
        print('clicked')
        status = self.cameraButton.isChecked()
        if status == True:
            self.cameraButton.setText('Click to Close Camera')
            
            # open the camera
            self.capture = VideoCapture()
            self.capture.change_pixmap_signal.connect(self.updateImage)
            self.capture.start()
            
        elif status == False:
            self.cameraButton.setText('Click to Open Camera')
            self.capture.stop()
            
    @qtc.pyqtSlot(np.ndarray)
    def updateImage(self,image_array):
        rgb_img = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
        h,w, ch = rgb_img.shape
        bytes_per_line = ch*w
        # convert into QImage
        convertedImage = qtg.QImage(rgb_img.data,w,h,bytes_per_line,qtg.QImage.Format_RGB888)
        scaledImage = convertedImage.scaled(600,600,qtc.Qt.KeepAspectRatio)
        qt_img = qtg.QPixmap.fromImage(scaledImage)

        # update to screen
        self.screen.setPixmap(qt_img)   
            
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = mainWindow()
    sys.exit(app.exec())