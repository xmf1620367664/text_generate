# _*_ coding:utf-8 _*_
"""
    Create by 南风木木  @2018/8/19
    主要实现功能：得到前面10个字符，预测第11个字符
    功能：给出11个字，拟出千字散文。（遇到句号为止）
"""
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

def train():
    training_path = 'wordstext'
    training_data=read_text(training_path)
    #training_data=''.join(str(training_data))
    forward_dict, reverse_dict = get_dictionary(training_path)
    num_of_words = len(forward_dict)
    print(num_of_words)


    n_inputs = 10
    n_hidden1 = 256
    n_hidden2 = 1024
    n_hidden3 = 512
    lr = 0.0001
    display_epoch=40
    save_epoch_num=1000
    training_epochs = 90000

    #定义输入输出占位符
    X = tf.placeholder(tf.float32, [None, n_inputs, 1])
    Y = tf.placeholder(tf.float32, [None, num_of_words])

    #获得预测结果
    pred=network(X,n_inputs,n_hidden1,n_hidden2,n_hidden3,num_of_words)
    softmax_pred=tf.nn.softmax(pred)
    loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))

    #梯度下降优化参数
    train_net=tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)

    #获取准确率
    acc_equal=tf.equal(tf.argmax(softmax_pred,1),tf.argmax(Y,1))
    acc=tf.reduce_mean(tf.cast(acc_equal,dtype=tf.float32))

    init=tf.global_variables_initializer()
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        counter=10
        # 重载保存的中间模型
        result = [0]
        # 查看模型状态
        ckpt = tf.train.get_checkpoint_state('./model')
        # 加载模型
        if ckpt and ckpt.model_checkpoint_path:
            print("model restoring")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            pattern = re.compile('\d+')
            result = pattern.findall(ckpt.model_checkpoint_path)
            print(result[0])
        #从十个文本中随机抽取10个文本，在10个文本中随机选取一个起始长度拿来做为输入，随机选取的起始长度的第11个字符为其正确输出
        for epoch in range(training_epochs-int(result[0])):
            #生成不重复的10个随机数
            random_list = []
            count=0
            sum_loss=0.0
            sum_acc=0.0
            while(count<counter):
                temp_Int = random.randint(0, 9)
                if temp_Int not in random_list:
                    random_list.append(temp_Int)
                    count+=1
            #获取文本信息的输入
            #获取随机起始位置
            #r_diss=random.randint(0, num_of_words - n_inputs - 2)
            #lambda r_diss:random.randint(0,len(training_data[r_int])-n_inputs-2)
            #random.seed(a=2)
            #input_X=[[forward_dict[training_data[r_int][i]] for i in range(random.randint(0,len(training_data[r_int])-n_inputs-2),random.randint(0,len(training_data[r_int])-n_inputs-2)+n_inputs)] for r_int in random_list]
            #random.seed(a=None)
            #获取模型的输入数据信息
            input_X=[]
            input_Y=[]
            for r_int in random_list:
                temp_Begin=random.randint(0,len(training_data[r_int])-n_inputs-2)
                input_X.append([forward_dict[training_data[r_int][i]] for i in range(temp_Begin,temp_Begin+n_inputs)])

                #生成下一个单词的one_hot编码信息
                one_hot=np.zeros([num_of_words],dtype=np.float32)
                #print(one_hot)
                #print(forward_dict[training_data[r_int][temp_Begin+n_inputs]])
                one_hot[forward_dict[training_data[r_int][temp_Begin+n_inputs]]]=1.0

                input_Y.append(one_hot)
                #print(one_hot)
            # 转换为占位符输入形状
            input_X = np.reshape(np.array(input_X), [-1, n_inputs, 1])
            input_Y = np.reshape(input_Y, [counter, num_of_words])

            #训练模型参数
            _,loss_,pred_,acc_=sess.run([train_net,loss,pred,acc],feed_dict={X:input_X,Y:input_Y})
            sum_loss+=loss_
            sum_acc+=acc_
            #显示模型信息
            if (epoch+1)  % display_epoch ==0:
                print("current epoch:{},avg_loss:{},avg_accuracy_rate:{}".format(epoch+1+int(result[0]),sum_loss/display_epoch,sum_acc/display_epoch))
                #获取当前输入的4*10个字符
                input_X=np.reshape(input_X,[-1,n_inputs])
                get_mess_X=[[reverse_dict[num] for num in col] for col in input_X]
                get_mess_X=np.reshape(np.array(get_mess_X),[counter,n_inputs])
                get_mess_Y=[reverse_dict[i] for i in  np.argmax(input_Y,1)]
                get_mess_Y_pre=[reverse_dict[i] for i in np.argmax(pred_,1)]
                print("字符预测:%s,实际结果:%s,预测结果:%s"%(get_mess_X,get_mess_Y,get_mess_Y_pre))
            if (epoch + 1) % save_epoch_num == 0:
                # .ckpt模型保存
                saver.save(sess, './model/model', global_step=epoch + 1 + int(result[0]))
                print("{} model had saved in ./model/model".format(epoch + 1 + int(result[0])))

if __name__ =="__main__":
    train()

