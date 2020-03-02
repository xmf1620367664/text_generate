# _*_ coding:utf-8 _*_
import sys
import generate
from PyQt5.QtWidgets import QApplication,QMainWindow

if __name__=='__main__':
    app=QApplication(sys.argv)
    mainWindow=QMainWindow()
    ui=generate.Ui_MainWindow()
    #向主窗口上添加控件
    ui.setupUi(mainWindow)
    #界面显示
    mainWindow.show()
    #系统循环，关闭退出
    sys.exit(app.exec_())