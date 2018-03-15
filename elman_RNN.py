""" 
coding: utf-8
@author: zhangxiang
"""
"""
在对脑电信号进行分类的时候，发现一篇文章对健康人，癫痫患者未发作时的脑电信号和癫痫发作时的脑电信号的分类使用了基于时序的
elman_RNN 神经网络进行建模，于是想在麻醉深度预测分类及其它时序相关的分类问题上使用这一模型。
"""
import numpy as np

class ELMAN_RNN(object):
    def __init__(self, input_num, hidden_num, output_num, learning_rate):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.learning_rate = learning_rate
        self.hidden_weights = np.random.random((self.input_num, self.hidden_num))
        self.output_weights = np.random.random((self.hidden_num, self.output_num))
        self.rnn_weights = np.random.random((self.hidden_num, self.hidden_num))
        self.hidden_bias = np.random.rand(1)
        self.output_bias = np.random.rand(1)
        self.hidden_output = np.zeros((1, self.hidden_num))

    def training(self, train_input, train_output):
        """training one time"""
        output = self.feed_forward(train_input)
        self.bptt(train_input, output, train_output)

    def calculate_the_cross_entropy(self, training_set):
        """get the total error loss"""
        loss = 0
        for i in range(np.array(training_set).shape[0]):
            x, y = training_set[i]
            y = np.array(y).reshape(1,2)
            result = self.feed_forward(x)
            loss += self.get_the_total_error(y, result)
        return loss

    def get_the_total_error(self, y, result):
        """loss = -∑yi*ln(ai), y is the real label, result is the softmax result"""
        loss = -np.sum(y*np.log(result))
        return loss

    def feed_forward(self, input):
        """calculate feed_forward value"""
        self.hidden_output = self.sigmoid(np.dot(np.array(input).reshape(1,2), self.hidden_weights) + np.dot(self.hidden_output, self.rnn_weights) + self.hidden_bias)
        return self.softmax(np.dot(self.hidden_output, self.output_weights) + self.output_bias)

    def bptt(self,input, output, train_output):
        """update the weights of all layers"""
        # claculate delta of output layers
        delta_of_output_layers = [0]*self.output_num
        for i in range(self.output_num):
            delta_of_output_layers[i] = self.calculate_output_wrt_rawout(output[0, i], train_output[i])

        # caculate delta of hidden layers
        delta_of_hidden_layers = [0]*self.hidden_num
        for i in range(self.hidden_num):
            d_error_wrt_hidden_output = 0.0
            for o in range(self.output_num):
                d_error_wrt_hidden_output += delta_of_output_layers[o]*self.output_weights[i, o]
            delta_of_hidden_layers[i] = d_error_wrt_hidden_output*self.calculate_output_wrt_netinput(self.hidden_output[0,i])

        # get the δw of output layers and update the weights
        for i in range(self.output_num):
            for weight_j in range(self.output_weights.shape[0]):
                delta_wrt_weight_j = delta_of_output_layers[i]*self.hidden_output[0,weight_j]

                self.output_weights[weight_j, i] -= self.learning_rate*delta_wrt_weight_j

        # get the δw of hidden layers and update the weights
        for i in range(self.hidden_num):
            for weight_j in range(self.hidden_weights.shape[0]):
                delta_wrt_weight_j = delta_of_hidden_layers[i]*input[weight_j]

                self.hidden_weights[weight_j, i] -= self.learning_rate*delta_wrt_weight_j

    def sigmoid(self, x):
        """activation function"""
        return 1.0/(1.0 + np.exp(-x))

    def softmax(self, x):
        """the activation for multiple output function"""
        return np.exp(x)/np.sum(np.exp(x))

    def calculate_output_wrt_rawout(self, output, train_output):
        """derivative of softmax function, actually in classification train_output equal to 1"""
        return (output - train_output)

    def calculate_output_wrt_netinput(self, output):
        """the derivative of sigmoid function"""
        return output*(1 - output)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    elman = ELMAN_RNN(input_num=2, hidden_num=4, output_num=2, learning_rate=0.02)
    train_x = [[1,2], [1,1], [1.5, 1.5], [2,1], [-1,-1], [-0.5, -0.5], [-1, -2], [-2, -1.5]]
    label_y = [[1,0], [1,0], [1,0], [1,0], [0,1], [0,1], [0,1], [0,1]]
    training_sets = [[[2,2],[1,0]], [[0.2, 0.8], [1,0]], [[-0.5, -0.8], [0, 1]], [[-1.2, -0.5], [0, 1]]]
    loss = []
    for i in range(1000):
        for x, y in zip(train_x, label_y):
            elman.training(x, y)
        loss.append(elman.calculate_the_cross_entropy(training_sets))
    plt.figure()
    plt.plot(loss)
    plt.title('the loss with the training')
    plt.show()
    print('training finished!')





