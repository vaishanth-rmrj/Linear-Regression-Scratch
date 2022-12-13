import numpy as np


class LinearRegressionModel:
    def __init__(self, learning_rate=1e-2, max_iter=500):
        self.weights = np.random.randn(2)
        self.max_iter = max_iter
        self.lr = learning_rate
        self.num_samples, self.num_features = 0, 0

    def fit(self, X, y):
        iteration = 0
        self.num_samples, self.num_features = X.shape
        
        while True:
            # updating the weights
            # using cost func's formula to compute the gradients wrt to W
            updated_weights = self.weights - self.lr * self.compute_grad(self.weights, X, y)

            if iteration % 80 == 1:
                loss = self.cost_func(updated_weights, X, y)
                print("Loss from cost function: {}, Weights: {}  Gradidents: {}".format(loss, updated_weights, self.compute_grad(self.weights, X, y)))
            
            # if the concurrent weigts are close stop updating
            # if (updated_weights[0]- self.weights[0])**2 + (updated_weights[1] - self.weights[1])**2 <= pow(10,-6):                
            #     return self.weights

            # if reached max iter stop updating
            if iteration > self.max_iter: 
                return self.weights

            iteration += 1
            self.weights = updated_weights               
    
    def predict(self, W, X):
        bias = self.weights[0]
        return (self.weights[1]*np.array(X[:,0]) + bias)

    def cost_func(self, W, X, y):
        y_pred = self.predict(W, X)

        # mean squared error
        cost = np.sum(np.square(y_pred - np.array(y))) / (2*self.num_samples)

        return cost
    
    def compute_grad(self, W, X, y):
        
        y_pred = self.predict(W, X)

        # gradients are computed by analytical method
        # J -> cost function
        # dJ/dw0 = 1/m * sum(y_pred - y_true)
        # dJ/dw1 = 1/m * sum(y_pred - y_true) * X   
        grad = [0, 0]
        grad[0] = (1/self.num_samples) * np.sum(y_pred - np.array(y))
        grad[1] = (1/self.num_samples) * np.sum((y_pred - np.array(y)) * np.array(X[:,0]))

        return np.array(grad)