# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Cifar_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(272, 352)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cifar1 = QtWidgets.QPushButton(self.centralwidget)
        self.cifar1.setGeometry(QtCore.QRect(60, 30, 151, 31))
        self.cifar1.setObjectName("cifar1")
        self.cifar2 = QtWidgets.QPushButton(self.centralwidget)
        self.cifar2.setGeometry(QtCore.QRect(60, 80, 151, 31))
        self.cifar2.setObjectName("cifar2")
        self.cifar3 = QtWidgets.QPushButton(self.centralwidget)
        self.cifar3.setGeometry(QtCore.QRect(60, 130, 151, 31))
        self.cifar3.setObjectName("cifar3")
        self.cifar4 = QtWidgets.QPushButton(self.centralwidget)
        self.cifar4.setGeometry(QtCore.QRect(60, 180, 151, 31))
        self.cifar4.setObjectName("cifar4")
        self.cifar5 = QtWidgets.QPushButton(self.centralwidget)
        self.cifar5.setGeometry(QtCore.QRect(60, 270, 151, 31))
        self.cifar5.setObjectName("cifar5")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setMaximum(9999)
        self.spinBox.setGeometry(QtCore.QRect(60, 230, 151, 22))
        self.spinBox.setObjectName("spinBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 272, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #action

        from Cifar10 import show_train_images,show_hyperparameters,show_accuracy,show_model_structure

        self.cifar1.clicked.connect(show_train_images)
        self.cifar2.clicked.connect(show_hyperparameters)
        self.cifar3.clicked.connect(show_model_structure)
        self.cifar4.clicked.connect(show_accuracy)
        self.cifar5.clicked.connect(self.test)


    def test(self):
        from Cifar10 import test_model

        x = self.spinBox.text()
        test_model(x)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cifar1.setText(_translate("MainWindow", "1.Show Train Images"))
        self.cifar2.setText(_translate("MainWindow", "2.Show Hyperparameters"))
        self.cifar3.setText(_translate("MainWindow", "3.Show Model Structure"))
        self.cifar4.setText(_translate("MainWindow", "4.Show Accuracy"))
        self.cifar5.setText(_translate("MainWindow", "5.Test"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

