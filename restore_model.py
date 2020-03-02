# _*_ coding:utf-8 _*_

import numpy as np
import tensorflow as tf
import collections
import random
import re

def read_text(text_name):
    """
    读取text_name文本中的数据，并转化为二维数组
    :param text_name:
    :return:
    """
    content=[]
    with open(text_name,encoding='utf-8') as file:
        for lines in file:
            content.append(lines)
    content_size=len(content)
    sentence=[[content[i][j] for j in range(len(content[i]))] for i in range(content_size)]
    #print(sentence)
    #sentence=np.array(sentence)
    #sentence=[i.tolist() for i in np.reshape(sentence,[constent_size,-1])]
    print(sentence)
    return sentence

def get_dictionary(text_name):
    """
    创建汉字字典,构建汉字与数字之间的映射关系
    :param text_name:
    :return:
    """
    words=read_text(text_name)
    #做类型转换
    # i_count=0
    # for i in words:
    #     temp_i=''.join(str(i))
    #     words[i_count]=temp_i
    #     i_count+=1
    words=''.join(str(words))
    print(words)
    #计数
    count_list=collections.Counter(words).most_common()
    count_list.append(('\u3000',2173))
    count_list.append(('\ue236',2174))
    count_list.append(('\ue364',2175))
    count_list.append(('\ue0ed',2176))
    count_list.append(('\ue010',2177))
    count_list.append(('\ue375',2178))
    count_list.append(('\ue362',2179))
    count_list.append(('\ue0e3',2180))
    count_list.append(('\ue256',2181))
    forward_dict={}
    reverse_dict={}
    k=0
    for key,_ in count_list:
        forward_dict[key]=k
        reverse_dict[k]=key
        k+=1
    """
        {"'": 0, ' ': 1, ',': 2, '，': 3, '的': 4, '。': 5, '是': 6, '一': 7, '我': 8, '不': 9, '了': 10, '有': 11, '“': 12, '”': 13,……
        {0: "'", 1: ' ', 2: ',', 3: '，', 4: '的', 5: '。', 6: '是', 7: '一', 8: '我', 9: '不', 10: '了', 11: '有', 12: '“', 13: '”',……
    """
    return forward_dict,reverse_dict

def network(x,n_inputs=10,n_hidden1=256,n_hidden2=1024,n_hidden3=512,num_of_words=2182):
    x=tf.reshape(x,shape=[-1,n_inputs])
    #时间结点划分
    x=tf.split(x,n_inputs,1)
    #构建网络细胞
    rnn_cell=tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(num_units=n_hidden1),
        tf.nn.rnn_cell.LSTMCell(num_units=n_hidden2),
        tf.nn.rnn_cell.LSTMCell(num_units=n_hidden3)
    ])

    outputs,state=tf.nn.static_rnn(rnn_cell,x,dtype=tf.float32)
    output=outputs[-1]
    #全连接层
    return tf.matmul(output,tf.get_variable('w',[n_hidden3,num_of_words],dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0.0,stddev=0.5)))+tf.ones([num_of_words],dtype=tf.float32)

def get_input(input_size=10):
    """
    获取输入信息并规格化信息
    :param input_size:
    :return:
    """
    input_data=input("请输入本篇文章的前十个字：")
    data_size=len(input_data)
    while data_size!=input_size:
        input_data = input("您的输入有误，请重新输入本篇文章的前十个字：")
        if len(input_data)==input_size:
            break
    sen_input=[i for i in input_data]
    #sen_input=np.reshape(np.array(sen_input),[-1,input_size,1])
    return sen_input

def predict(input_string):
    sen_input=input_string
    # print(sen_input)
    # print(type(sen_input))
    # exit()
    # 定义源数据的路径
    training_path = 'wordstext'
    # training_data = read_text(training_path)
    # training_data=''.join(str(training_data))
    # 获取正向字典和反向字典
    forward_dict, reverse_dict = get_dictionary(training_path)
    # 获取字典长度
    num_of_words = len(forward_dict)
    print(num_of_words)

    print(sen_input)
    n_inputs = 10
    n_hidden1 = 256
    n_hidden2 = 1024
    n_hidden3 = 512
    # 将输入的信息进行转化
    print(sen_input)
    sen_input = [forward_dict[i] for i in sen_input]
    # 获取输入的数字序列信息
    result = sen_input
    # reshape为占位符输入的格式
    sen_input = np.reshape(np.array(sen_input), [-1, n_inputs, 1])
    # 定义输入输出占位符
    X = tf.placeholder(tf.float32, [None, n_inputs, 1])
    # 获得预测结果
    pred = network(X, n_inputs, n_hidden1, n_hidden2, n_hidden3, num_of_words)
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
        while len(result) <= 100:
            sen_input = result[-n_inputs:]
            sen_input = np.reshape(np.array(sen_input), [-1, n_inputs, 1])
            result.append(sess.run(tf.argmax(tf.nn.softmax(sess.run(pred, feed_dict={X: sen_input})), 1))[0])
            print(result)
        result.append(forward_dict['。'])
        print(result)
        with open('result_text.txt', 'w+', encoding='utf-8') as file:
            [file.write(reverse_dict[i]) for i in result]
        dict_result=[reverse_dict[i] for i in result]
        dict_result=''.join(dict_result)
    return dict_result
if __name__ =="__main__":
    #获取输入的数据
    sen_input=get_input(input_size=10)
    print(sen_input)
    print(type(sen_input))
    exit()
    #定义源数据的路径
    training_path = 'wordstext'
    #training_data = read_text(training_path)
    # training_data=''.join(str(training_data))
    #获取正向字典和反向字典
    forward_dict, reverse_dict = get_dictionary(training_path)
    #获取字典长度
    num_of_words = len(forward_dict)
    print(num_of_words)

    print(sen_input)
    n_inputs = 10
    n_hidden1 = 256
    n_hidden2 = 1024
    n_hidden3 = 512
    # 将输入的信息进行转化
    print(sen_input)
    sen_input = [forward_dict[i] for i in sen_input]
    #获取输入的数字序列信息
    result=sen_input
    #reshape为占位符输入的格式
    sen_input = np.reshape(np.array(sen_input), [-1, n_inputs, 1])
    # 定义输入输出占位符
    X = tf.placeholder(tf.float32, [None, n_inputs, 1])
    # 获得预测结果
    pred = network(X, n_inputs, n_hidden1, n_hidden2, n_hidden3, num_of_words)
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
        #or forward_dict['。'] !=sess.run(tf.argmax(tf.nn.softmax(sess.run(pred,feed_dict={X:sen_input})),1))[0]
        while len(result)<=2000:
            sen_input=result[-n_inputs:]
            sen_input = np.reshape(np.array(sen_input), [-1, n_inputs, 1])
            result.append(sess.run(tf.argmax(tf.nn.softmax(sess.run(pred,feed_dict={X:sen_input})),1))[0])
            print(result)
        result.append(forward_dict['。'])
        print(result)
        with open('result_text.txt','w+',encoding='utf-8') as file:
            [file.write(reverse_dict[i]) for i in result]