'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate = 1, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.regLambda = 0.001

    def sigmoid(self, value):

        return 1.0 / (1 + np.exp(-value))

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy arrayde
        '''
        n, d = X.shape
        self.targets = np.unique(y)
        self.targets_number = len(self.targets)
        self.layer_mark = np.zeros(len(self.layers)+1)
        np.random.seed(15)
        elements_count = 0
        #self.theta = []
        self.theta_list = []
        self.delta_list = []
        if len(self.layers) == 1:
            elements = (self.layers[0] * (d+1))
            elements_count += elements
            self.layer_mark[0] = elements_count
            temp = np.ones((self.layers[0], (d + 1)))

            for x in np.nditer(temp, op_flags=['readwrite']):
                x[...] = np.random.uniform(-self.epsilon, self.epsilon)
          
            self.theta_list.append(temp)
            self.delta_list.append(np.zeros((int(self.layers[0]), (d + 1))))

            elements = (self.targets_number * (self.layers[0]+1))
            elements_count += elements
            self.layer_mark[1] = elements_count
            temp = np.ones((self.targets_number, (self.layers[0]+1)))

            for x in np.nditer(temp, op_flags=['readwrite']):
                x[...] = np.random.uniform(-self.epsilon, self.epsilon)
          
            self.theta_list.append(temp)
            self.delta_list.append(np.zeros((self.targets_number, (self.layers[0]+1))))

        else:
            for i in range(len(self.layers)):
                if i == 0:
                    elements = (self.layers[i] * (d+1))
                    temp = np.ones((self.layers[0], (d + 1)))
                    for x in np.nditer(temp, op_flags=['readwrite']):
                        x[...] = np.random.uniform(-self.epsilon, self.epsilon)
          
                    self.theta_list.append(temp)
                    self.delta_list.append(np.zeros((self.layers[0], d+1)))
                elif i == (len(self.layers)-1):
                    elements = (self.targets_number * (self.layers[i]+1))
                    temp = np.ones((self.targets_number , (self.layers[i]+1)))
                    for x in np.nditer(temp, op_flags=['readwrite']):
                        x[...] = np.random.uniform(-self.epsilon, self.epsilon)
          
                    self.theta_list.append(temp)
                    self.delta_list.append(np.zeros((self.targets_number * (self.layers[i]+1))))
                else:
                    elements = (self.layers[i] * (self.layers[i-1]+1))
                    temp = np.ones((self.layers[0], (d + 1)))
                    for x in np.nditer(temp, op_flags=['readwrite']):
                        x[...] = np.random.uniform(-self.epsilon, self.epsilon)
          
                    self.theta_list.append(temp)
                    self.delta_list.append(np.zeros((self.layers[i],(self.layers[i-1]+1))))

                elements_count += elements
                self.layer_mark[i] = elements_count

        for epoch in range(self.numEpochs):
            print epoch , ' / ' , self.numEpochs

            for i in range(n):
                data = X[i,:]
                target = y[i]
                a_list = self.forwardPropagation(X[i])

                err_list = self.backwardPropagation(y[i], a_list)

                for k in range(len(self.delta_list)):                    

                    biasdx = np.concatenate((np.ones((1, 1)), a_list[k].T), axis=1)
                    self.delta_list[k] = self.delta_list[k] + np.dot(err_list[k], biasdx)

            for k in range(len(self.delta_list)):

                t = self.theta_list[k]
                self.delta_list[k] = (1.0 / n) * self.delta_list[k] + self.regLambda * self.theta_list[k]
                t = t - self.learningRate*self.delta_list[k]
                self.theta_list[k] = t


    def forwardPropagation(self, instance):
        #generate theta list
        result = []
        #loop
        instance = np.reshape(instance, (len(instance), 1))
        result.append(instance)
        a = np.insert(instance, 0, 1)

        z = np.dot(self.theta_list[0], a)
        a = self.sigmoid(z)
        a = np.reshape(a, (len(a), 1))
        if len(self.layers) == 1:
            result.append(a)
            a = np.insert(a, 0, 1)            
            z = np.dot(self.theta_list[1], a)
            a = self.sigmoid(z)
            a = np.reshape(a, (len(a), 1))

        else:
            for i in range(len(self.layers)):
                result.append(a)
                a = np.insert(a, 0, 1)
                z = np.dot(self.theta_list[i+1], a)
                a = self.sigmoid(z)
                a = np.reshape(a, (len(a), 1))

        result.append(a)
        return result

    def backwardPropagation(self, target, alist):
        #result = errlist
        errorList = []

        theta_list_reverse = self.theta_list[::-1]


        alist_reverse = alist[::-1]
        outputLayer = alist_reverse[0]
        #print self.setStandard(target)
        errorList.append(outputLayer - self.setStandard(target))

        for i in range(len(self.layers)):
            
            dg = alist_reverse[i+1]*(1-alist_reverse[i+1])

            currentError = np.dot((theta_list_reverse[i][:,1:]).T, errorList[i])*dg

            errorList.append(currentError)
        result = errorList[::-1]
        return result

    def setStandard(self, y):
        temp = np.zeros((self.targets_number,1))
        for k in range(self.targets_number):
            if self.targets[k] == y:                
                temp[k,0] = 1
        return temp


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d = X.shape

        output = np.zeros(n)
        for i in range(n):
            result = self.forwardPropagation(X[i, :])[-1]

            output[i] = self.targets[np.argmax(result)]

        return output


    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''