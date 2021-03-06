'''
Logistic Regression 

Group members: Nie-Binbin  Ma-Shicheng  Cao-Liyu

'''


import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.w = np.random.rand(np.shape(X_train)[1]).reshape((np.shape(X_train)[1], 1))
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.reshape((len(y_train), 1))
        self.y_test = y_test.reshape((len(y_test), 1))

    # def sigmoid(self, z):
    #     s = 1.0 / (1 + np.exp(-z))
    #     return s

    def loss_function(self):  # we want to minimize the loss function
        res = np.dot(self.X_train, self.w) * \
            self.y_train.reshape((len(self.y_train), 1))
        res -= np.log(1 + np.exp(np.dot(self.X_train, self.w)))
        res = -np.sum(res)
        return res

    def gradient(self):  # witten by NBB
        res = self.y_train - np.exp(np.dot(self.X_train, self.w))/(1+np.exp(np.dot(self.X_train, self.w)))
        res = - np.sum(res*self.X_train, axis=0)
        res = res.reshape((-1,1))
        return res

    # def gradient(self):  # witten by WXY 
    #     res = self.X_train.T.dot(1-1/(1+np.exp(self.X_train.dot(self.w)))-self.y_train)
    #     return res

        


    def predict(self, threshold = 0.5):
        y = 1 / np.exp(np.dot(self.X_test, self.w))
        predict_result = np.zeros((len(self.y_test), 1))
        # print(y)
        for i in range(len(predict_result)):
            if (y[i] >= threshold):
                predict_result[i] = 0
            else:
                predict_result[i] = 1
        acc = 1 - np.sum(abs(predict_result - self.y_test))/len(self.y_test)
        
        return acc

    def propagate(self, N, eta = 0.00001, eps = 10**(-6)):
        # N is max iteration times
        loss_arr = np.array([0])  
        while True and len(loss_arr) < N:
            self.w = self.w - eta*self.gradient()
            loss_arr = np.append(loss_arr, self.loss_function())
            print('Loss function value:{}'.format(self.loss_function()))
            if (abs(abs(loss_arr[-1]) - abs(loss_arr[-2])) < eps):
                print('break')
                break
        plt.figure()
        acc = self.predict()
        plt.plot(loss_arr[1:],linewidth = 2,label = 'classification accuracy'+str(acc*100)+'%')
        plt.xlabel('iteration')
        plt.ylabel('Loss function')
        plt.title('convergence curve')
        plt.legend()
        
    def confuseMatrix(self):
        
        y = 1 / np.exp(np.dot(self.X_test, self.w))
        predict_result = np.zeros((len(self.y_test), 1))
        # print(y)
        for i in range(len(predict_result)):
            if (y[i] >= 0.5):
                predict_result[i] = 0
            else:
                predict_result[i] = 1
        confuse_matrix = np.zeros((2,2))
        for i in range(len(self.y_test)):
            if (self.y_test[i] ==0 and predict_result[i]==0):
                confuse_matrix[0,0] += 1
            elif (self.y_test[i] ==1 and predict_result[i]==1):
                confuse_matrix[1,1] += 1
            elif (self.y_test[i] ==0 and predict_result[i]==1):
                confuse_matrix[0,1] +=1
            elif (self.y_test[i] ==1 and predict_result[i]==0):
                confuse_matrix[1,0] +=1
        plt.figure()
        plt.imshow(confuse_matrix,cmap = 'Blues')
        indices = range(len(confuse_matrix))
        plt.xticks(indices,[0,1])
        plt.yticks(indices,[0,1])
        plt.colorbar()
        plt.xlabel('Real value')
        plt.ylabel('Predict value')
        plt.title('Confused Matrix')
        for first_index in range(len(confuse_matrix)):    #?????????
            for second_index in range(len(confuse_matrix)):    #?????????
                plt.text(first_index, second_index, confuse_matrix[first_index][second_index])
            

