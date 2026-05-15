import numpy as np


class losso_reggrsion():

    def __init__(self,learing_rate,no_of_itera,lambda_val):
        self.learing_rate=learing_rate
        self.no_of_itera=no_of_itera
        self.lambda_val=lambda_val

    def fit(self,x,y):
        self.m,self.n=x.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.x=x
        self.y=y
        for i in  range(self.no_of_itera):
            self.updates_weights()

    def updates_weights(self):
        y_predic=self.predict(self.x)
        dw=np.zeros(self.n)
        for i in range(self.n):
            if(self.w[i]>0):
                dw[i]=(-2*(self.x[:,i].dot(self.y-y_predic)+self.lambda_val))/self.m
            else:
                dw=(-2*self.x[:,i].dot(self.y-y_predic)-self.lambda_val)/self.m

        db=(-2*np.sum(self.y-y_predic))/self.m
        w=w-self.learing_rate*dw
        b=b-self.learing_rate*db

    def predict(self,X):
        return X.dot(self.w)+self.b