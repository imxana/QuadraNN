# Q3. 自定义神经网络，使用训练集 y=2.7x^2+3.5x-11.6 进行训练

import numpy as np
import sys
from matplotlib import pyplot as plt


######################################################################################################################

#实现一个加法层
class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):      #f(x,y)=x+y
        out = x + y
        return out
    def backward(self,dout):    #f'(x)=1*dout (dout的含义为上游导数)
        dx = dout * 1
        dy = dout * 1
        return dx,dy
    
#实现一个乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x,y):      #f(x,y)=x*y
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self,dout):    #f'(x)=dout*y
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy
    
#实现一个e的指数层 
class ExpLayer:
    def __init__(self):
        pass
    def forward(self,x):        # f(x)=e^x  
        self.x = np.exp(x)
        return self.x
    def backward(self,dout):    #f'(x)=dout*e^x
        return dout * self.x
    
#实现了倒数层，是倒数
class ReciprocalLayer:
    def __init__(self):
        pass
    def forward(self,x):        #f(x)=1/x
        self.x =  1/x
        return self.x
    def backward(self,dout):    #f'(x)=-dout*(1/x^2)
        return -dout *self.x * self.x
    
class Sigmoid:
    def __init__(self):
        pass
    def forward(self,x):        #sigmoid(x)
        self.layer0 = MulLayer()  
        xx = self.layer0.forward(x, -1)
        self.layer1 = ExpLayer()
        xx = self.layer1.forward(xx)
        self.layer2 = AddLayer()
        xx = self.layer2.forward(xx, 1)
        self.layer3 = ReciprocalLayer()
        xx = self.layer3.forward(xx)
        self.a = xx
        return xx
    def backward(self,dout):    #sigmoid'(x)
        x = self.layer3.backward(dout)
        x, y = self.layer2.backward(x)
        x = self.layer1.backward(x)
        x, y = self.layer0.backward(x)
        return x

######################################################################################################################

# 简化实现class Sigmoid 2023/5/24 by xana
class SigmoidLayer:
    def __init__(self):
        pass
    def forward(self, x):
        self.e = np.exp(-x)     # sigmoid公式：f(x)=1/(1+e^-x)，保存下对应变量方便求导
        self.r = 1/(self.e + 1)
        return self.r
    def backward(self,dout):    # sigmoid求导
        return dout * self.r * self.r * self.e

 
# 构建全连接层
class DenseLayer:
    def __init__(self,input_dim,hidden_nodes,lr):
        self.w = np.random.normal(0.0, 0.5, size=(input_dim, hidden_nodes))
        self.b = np.zeros(shape=(1,hidden_nodes))
        self.lr = lr
        self.x = None
        self.dw = None
        self.db = None
    def forward(self,x):
        self.x = x
        out = np.matmul(x,self.w) + self.b
        return out
    def backward(self,dout):
        dx = np.matmul(dout,self.w.T)
        nums = dout.shape[0]
        self.dw = np.matmul(self.x.T, dout) / nums
        self.db = np.mean(dout,axis=0)
        self.w = self.w - self.lr * self.dw
        self.b = self.b - self.lr * self.db
        return dx

#构建激活函数层
class ActivationLayer:
    # param:activatetype is some activatefunction,e.g:sigmoid,relu...
    def __init__(self, activateType):
        self.activate = activateType()
    def forward(self,x):
        xx = self.activate.forward(x)
        return xx
    def backward(self, dout):
        return self.activate.backward(dout)

# 构建序列类
class Sequential:
    def __init__(self):
        self.layers = []               
        self.layersLevel = 0
    def add(self,layer):
        self.layers.append(layer)
        self.layersLevel += 1
    def fit(self,x_data,y_data,epoch):          #使用训练数据进行迭代出loss1值后，进行反向传播修正
        for j in range(epoch):
            x = x_data
            for i in range(self.layersLevel):
                x = self.layers[i].forward(x)
            loss1 = x - y_data
            for i in range(self.layersLevel-1): #反向传播，向下迭代loss值
                loss1 = self.layers[self.layersLevel - i - 1].backward(loss1)
            loss = 0.5 * np.mean(np.square(x - y_data)) # 吐槽下- -，虚假的loss值:MSE，真实的loss值:(pred-real)
            self.view_loss(j + 1, epoch, loss)
    def view_loss(self, step, total, loss):     # 打印当前进度
        rate = step / total
        rate_num = int(rate * 40)
        s1, s2 = '>' * rate_num, '-' * (40 - rate_num)
        r = f'\rstep:{step} loss:{loss:.4f}[{s1}{s2}] {int(step*100/total)}%               '
        sys.stdout.write(r)
        sys.stdout.flush()
    def predict(self,x_data):
        x = x_data
        for i in range(self.layersLevel):
            x = self.layers[i].forward(x)
        return x
    
class Task():
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a,b,c
    def Run(self):
        a,b,c = self.a, self.b, self.c
        m = -b/a/2
        x_data = np.array([[x] for x in np.arange(-10+m, 10+m, 0.1)])
        y_data = a*np.square(x_data) + b*x_data + c # 10*np.sin(x_data)

        input_size = 1
        hidden1_size = 5
        hidden2_size = 10
        output_size = 1
        learning_rate = 0.01
        epochs = 50000

        # 这个模型的曲线预测
        model = Sequential()
        model.add(DenseLayer(input_size, hidden1_size, lr=learning_rate))
        model.add(SigmoidLayer())  # model.add(ActivationLayer(Sigmoid))
        model.add(DenseLayer(hidden1_size, hidden2_size, lr=learning_rate))
        model.add(SigmoidLayer()) # model.add(ActivationLayer(Sigmoid))
        model.add(DenseLayer(hidden2_size, output_size, lr=learning_rate))
        model.fit(x_data, y_data, epochs)

        self.model = model
        self.x_data = x_data
        self.y_data = y_data

    def Show(self):
        y_preData = self.model.predict(self.x_data)
        
        plt.plot(self.x_data, self.y_data, label="real", linestyle = "--")
        plt.plot(self.x_data, y_preData, label="predict")
        plt.title(f"Predict Line: a={self.a} b={self.b} c={self.c}")
        plt.legend()
        plt.show()

    def Save(self):
        np.save("QuadraNN_model.npy", np.asanyarray([self]))

if __name__ == '__main__':
    task = Task(2.7, 3.5, -11.6)
    task.Run()
    task.Show()
    task.Save()

    # task = np.load("QuadraNN_model.npy", allow_pickle=True)[0]
    # task.Show()