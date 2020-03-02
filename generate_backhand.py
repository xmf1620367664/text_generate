# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'generate.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from restore_model import predict
import numpy as np
import tensorflow as tf
import collections
import time


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 0, 211, 81))
        self.label.setStyleSheet("\n"
"font: 20pt \"DFHaiBaoW12-GB5\";")
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(30, 80, 521, 471))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        self.calendarWidget = QtWidgets.QCalendarWidget(self.centralwidget)
        self.calendarWidget.setGeometry(QtCore.QRect(560, 190, 231, 197))
        self.calendarWidget.setObjectName("calendarWidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(580, 420, 184, 99))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setText("")
        self.lineEdit.setMaxLength(10)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout.addWidget(self.lineEdit)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.progressBar = QtWidgets.QProgressBar(self.widget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(self.lineEdit.clear)
        self.pushButton_2.clicked.connect(self.textEdit.clear)
        self.pushButton_2.clicked.connect(self.progressBar.reset)
        self.pushButton.clicked.connect(self.set_string)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def read_text(self,text_name):
        """
        读取text_name文本中的数据，并转化为二维数组
        :param text_name:
        :return:
        """
        content = []
        with open(text_name, encoding='utf-8') as file:
            for lines in file:
                content.append(lines)
        content_size = len(content)
        sentence = [[content[i][j] for j in range(len(content[i]))] for i in range(content_size)]
        # print(sentence)
        # sentence=np.array(sentence)
        # sentence=[i.tolist() for i in np.reshape(sentence,[constent_size,-1])]
        print(sentence)
        return sentence

    def get_dictionary(self,text_name):
        """
        创建汉字字典,构建汉字与数字之间的映射关系
        :param text_name:
        :return:
        """
        words = self.read_text(text_name)
        # 做类型转换
        # i_count=0
        # for i in words:
        #     temp_i=''.join(str(i))
        #     words[i_count]=temp_i
        #     i_count+=1
        words = ''.join(str(words))
        print(words)
        # 计数
        count_list = collections.Counter(words).most_common()
        count_list.append(('\u3000', 2173))
        count_list.append(('\ue236', 2174))
        count_list.append(('\ue364', 2175))
        count_list.append(('\ue0ed', 2176))
        count_list.append(('\ue010', 2177))
        count_list.append(('\ue375', 2178))
        count_list.append(('\ue362', 2179))
        count_list.append(('\ue0e3', 2180))
        count_list.append(('\ue256', 2181))
        forward_dict = {}
        reverse_dict = {}
        k = 0
        for key, _ in count_list:
            forward_dict[key] = k
            reverse_dict[k] = key
            k += 1
        """
            {"'": 0, ' ': 1, ',': 2, '，': 3, '的': 4, '。': 5, '是': 6, '一': 7, '我': 8, '不': 9, '了': 10, '有': 11, '“': 12, '”': 13,……
            {0: "'", 1: ' ', 2: ',', 3: '，', 4: '的', 5: '。', 6: '是', 7: '一', 8: '我', 9: '不', 10: '了', 11: '有', 12: '“', 13: '”',……
        """
        return forward_dict, reverse_dict

    def network(self,x, n_inputs=10, n_hidden1=256, n_hidden2=1024, n_hidden3=512, num_of_words=2182):
        x = tf.reshape(x, shape=[-1, n_inputs])
        # 时间结点划分
        x = tf.split(x, n_inputs, 1)
        # 构建网络细胞
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.LSTMCell(num_units=n_hidden1),
            tf.nn.rnn_cell.LSTMCell(num_units=n_hidden2),
            tf.nn.rnn_cell.LSTMCell(num_units=n_hidden3)
        ])

        outputs, state = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
        output = outputs[-1]
        # 全连接层
        return tf.matmul(output, tf.get_variable('w', [n_hidden3, num_of_words], dtype=tf.float32,
                                                 initializer=tf.random_normal_initializer(mean=0.0,
                                                                                          stddev=0.5))) + tf.ones(
            [num_of_words], dtype=tf.float32)

    def set_string(self):
        string_len=100
        input_string=self.lineEdit.text()
        sen_input = [i for i in input_string]
        # #print(sen_input)
        #
        # result_string=predict(sen_input)
        # #print(result_string)
        # self.textEdit.setText(result_string)

        # 定义源数据的路径
        training_path = 'wordstext'
        # training_data = read_text(training_path)
        # training_data=''.join(str(training_data))
        # 获取正向字典和反向字典
        forward_dict, reverse_dict = self.get_dictionary(training_path)
        # 获取字典长度
        num_of_words = len(forward_dict)
        #print(num_of_words)

        #print(sen_input)
        n_inputs = 10
        n_hidden1 = 256
        n_hidden2 = 1024
        n_hidden3 = 512
        # 将输入的信息进行转化
        #print(sen_input)
        sen_input = [forward_dict[i] for i in sen_input]
        # 获取输入的数字序列信息
        result = sen_input
        # reshape为占位符输入的格式
        sen_input = np.reshape(np.array(sen_input), [-1, n_inputs, 1])
        # 定义输入输出占位符
        X = tf.placeholder(tf.float32, [None, n_inputs, 1])
        # 获得预测结果
        pred = self.network(X, n_inputs, n_hidden1, n_hidden2, n_hidden3, num_of_words)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 查看模型状态
            ckpt = tf.train.get_checkpoint_state('./model')
            # 加载模型
            if ckpt and ckpt.model_checkpoint_path:
                print("model restoring")
                saver.restore(sess, ckpt.model_checkpoint_path)
            print(result)
            print(forward_dict['。'])
            # or forward_dict['。'] !=sess.run(tf.argmax(tf.nn.softmax(sess.run(pred,feed_dict={X:sen_input})),1))[0]
            while len(result) <= string_len:
                sen_input = result[-n_inputs:]
                sen_input = np.reshape(np.array(sen_input), [-1, n_inputs, 1])
                result.append(sess.run(tf.argmax(tf.nn.softmax(sess.run(pred, feed_dict={X: sen_input})), 1))[0])
                #print(result)
                dict_result = [reverse_dict[i] for i in result]
                self.textEdit.setText(''.join(dict_result))
                process=(len(result))*100/string_len
                self.progressBar.setValue(process)
                # 实时刷新界面
                QApplication.processEvents()
                time.sleep(0.5)
            result.append(forward_dict['。'])
            #print(result)
            with open('result_text.txt', 'w+', encoding='utf-8') as file:
                [file.write(reverse_dict[i]) for i in result]
            dict_result = [reverse_dict[i] for i in result]
            dict_result = ''.join(dict_result)
            self.textEdit.setText(''.join(dict_result))





    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" color:#00ff8c;\">文章生成器</span></p></body></html>"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "文章生成总领(限十个字符长度)："))
        self.pushButton_2.setText(_translate("MainWindow", "重置"))
        self.pushButton.setText(_translate("MainWindow", "提交"))
