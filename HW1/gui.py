# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_opencvdl_hw1(object):
    def setupUi(self, opencvdl_hw1):
        opencvdl_hw1.setObjectName("opencvdl_hw1")
        opencvdl_hw1.resize(760, 348)
        self.centralwidget = QtWidgets.QWidget(opencvdl_hw1)
        self.centralwidget.setObjectName("centralwidget")
        self.image_processing_1 = QtWidgets.QPushButton(self.centralwidget)
        self.image_processing_1.setGeometry(QtCore.QRect(40, 60, 101, 31))
        self.image_processing_1.setObjectName("image_processing_1")
        self.image_processing_2 = QtWidgets.QPushButton(self.centralwidget)
        self.image_processing_2.setGeometry(QtCore.QRect(40, 120, 101, 31))
        self.image_processing_2.setObjectName("image_processing_2")
        self.image_processing_3 = QtWidgets.QPushButton(self.centralwidget)
        self.image_processing_3.setGeometry(QtCore.QRect(40, 180, 101, 31))
        self.image_processing_3.setObjectName("image_processing_3")
        self.image_processing_4 = QtWidgets.QPushButton(self.centralwidget)
        self.image_processing_4.setGeometry(QtCore.QRect(40, 240, 101, 31))
        self.image_processing_4.setObjectName("image_processing_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 30, 101, 20))
        self.label.setObjectName("label")
        self.image_smoothing_1 = QtWidgets.QPushButton(self.centralwidget)
        self.image_smoothing_1.setGeometry(QtCore.QRect(200, 80, 101, 31))
        self.image_smoothing_1.setObjectName("image_smoothing_1")
        self.image_smoothing_2 = QtWidgets.QPushButton(self.centralwidget)
        self.image_smoothing_2.setGeometry(QtCore.QRect(200, 150, 101, 31))
        self.image_smoothing_2.setObjectName("image_smoothing_2")
        self.image_smoothing_3 = QtWidgets.QPushButton(self.centralwidget)
        self.image_smoothing_3.setGeometry(QtCore.QRect(200, 220, 101, 31))
        self.image_smoothing_3.setObjectName("image_smoothing_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 30, 101, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(370, 30, 101, 20))
        self.label_3.setObjectName("label_3")
        self.edge_detection_1 = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection_1.setGeometry(QtCore.QRect(360, 60, 101, 31))
        self.edge_detection_1.setObjectName("edge_detection_1")
        self.edge_detection_2 = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection_2.setGeometry(QtCore.QRect(360, 120, 101, 31))
        self.edge_detection_2.setObjectName("edge_detection_2")
        self.edge_detection_3 = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection_3.setGeometry(QtCore.QRect(360, 180, 101, 31))
        self.edge_detection_3.setObjectName("edge_detection_3")
        self.edge_detection_4 = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection_4.setGeometry(QtCore.QRect(360, 240, 101, 31))
        self.edge_detection_4.setObjectName("edge_detection_4")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(520, 30, 101, 20))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(520, 70, 51, 20))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(520, 120, 51, 20))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(520, 220, 51, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(520, 170, 51, 20))
        self.label_8.setObjectName("label_8")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(570, 70, 113, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(570, 120, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(570, 170, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(570, 220, 113, 20))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(700, 70, 51, 20))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(700, 170, 51, 20))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(700, 220, 51, 20))
        self.label_11.setObjectName("label_11")
        self.transformation = QtWidgets.QPushButton(self.centralwidget)
        self.transformation.setGeometry(QtCore.QRect(530, 260, 181, 23))
        self.transformation.setObjectName("transformation")
        opencvdl_hw1.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(opencvdl_hw1)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 760, 21))
        self.menubar.setObjectName("menubar")
        opencvdl_hw1.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(opencvdl_hw1)
        self.statusbar.setObjectName("statusbar")
        opencvdl_hw1.setStatusBar(self.statusbar)

        self.retranslateUi(opencvdl_hw1)
        QtCore.QMetaObject.connectSlotsByName(opencvdl_hw1)

        #action

        from Image_Processing import load_image,color_separation,image_flipping,blending
        from Image_Smoothing import median_filter,gaussian_blur,bilateral_filter
        from Edge_Detection import gaussian_filter,sobel_x,sobel_y,magnitude

        self.image_processing_1.clicked.connect(load_image)
        self.image_processing_2.clicked.connect(color_separation)
        self.image_processing_3.clicked.connect(image_flipping)
        self.image_processing_4.clicked.connect(blending)

        self.image_smoothing_1.clicked.connect(median_filter)
        self.image_smoothing_2.clicked.connect(gaussian_blur)
        self.image_smoothing_3.clicked.connect(bilateral_filter)

        self.edge_detection_1.clicked.connect(gaussian_filter)
        self.edge_detection_2.clicked.connect(sobel_x)
        self.edge_detection_3.clicked.connect(sobel_y)
        self.edge_detection_4.clicked.connect(magnitude)

        self.transformation.clicked.connect(self.trans)


    def trans(self):
        from Transformation import transformation_
        r=self.lineEdit.text()
        s=self.lineEdit_2.text()
        x=self.lineEdit_3.text()
        y=self.lineEdit_4.text()
        if r=='':
            r=0
        if s=='':
            s=1
        if x=='':
            x=0
        if y=='':
            y=0
        transformation_(r,s,x,y)


    def retranslateUi(self, opencvdl_hw1):
        _translate = QtCore.QCoreApplication.translate
        opencvdl_hw1.setWindowTitle(_translate("opencvdl_hw1", "2020 Opencvdl HW1"))
        self.image_processing_1.setText(_translate("opencvdl_hw1", "1.1 Load Image"))
        self.image_processing_2.setText(_translate("opencvdl_hw1", "1.2 Color seperation"))
        self.image_processing_3.setText(_translate("opencvdl_hw1", "1.3 Image Flipping"))
        self.image_processing_4.setText(_translate("opencvdl_hw1", "1.4Blending"))
        self.label.setText(_translate("opencvdl_hw1", "1.Image Processing"))
        self.image_smoothing_1.setText(_translate("opencvdl_hw1", "2.1 Median Filter"))
        self.image_smoothing_2.setText(_translate("opencvdl_hw1", "2.2 Gaussian Blur"))
        self.image_smoothing_3.setText(_translate("opencvdl_hw1", "2.3 Bilateral Filter"))
        self.label_2.setText(_translate("opencvdl_hw1", "2.Image Smoothing"))
        self.label_3.setText(_translate("opencvdl_hw1", "3.Edge Detection"))
        self.edge_detection_1.setText(_translate("opencvdl_hw1", "3.1 Gaussian Blur"))
        self.edge_detection_2.setText(_translate("opencvdl_hw1", "3.2 Sobel X"))
        self.edge_detection_3.setText(_translate("opencvdl_hw1", "3.3 Sobel Y"))
        self.edge_detection_4.setText(_translate("opencvdl_hw1", "3.4 Magnitude"))
        self.label_4.setText(_translate("opencvdl_hw1", "4.Transformation"))
        self.label_5.setText(_translate("opencvdl_hw1", "Rotation :"))
        self.label_6.setText(_translate("opencvdl_hw1", "Scaling :"))
        self.label_7.setText(_translate("opencvdl_hw1", "Ty :"))
        self.label_8.setText(_translate("opencvdl_hw1", "Tx :"))
        self.label_9.setText(_translate("opencvdl_hw1", "deg"))
        self.label_10.setText(_translate("opencvdl_hw1", "pixel"))
        self.label_11.setText(_translate("opencvdl_hw1", "pixel"))
        self.transformation.setText(_translate("opencvdl_hw1", "4.Transformation"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    opencvdl_hw1 = QtWidgets.QMainWindow()
    ui = Ui_opencvdl_hw1()
    ui.setupUi(opencvdl_hw1)
    opencvdl_hw1.show()
    sys.exit(app.exec_())