#imports
from models.decisiontrees import DecisionTreeClassifier
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.base import clone

## boosting classifier ##
class GradientBoostTreeClassifier(object):
    #initializer
    def __init__(self,  n_elements : int = 100, learning_rate : float = 0.01, record_training_f1 : bool = False) -> None:
        self.weak_learner  = DecisionTreeClassifier(max_depth=5)
        self.n_elements    = n_elements
        self.learning_rate = learning_rate
        self.f             = []
        self.model_weights = []
        self.f1s           = []
        self.record_f1s    = record_training_f1
        
    #destructor
    def __del__(self) -> None:
        del self.weak_learner
        del self.n_elements
        del self.learning_rate
        del self.f
        del self.model_weights
        del self.f1s
    
    #public function to return model parameters
    def get_params(self, deep : bool = False) -> Dict:
        return {'weak_learner':self.weak_learner,'n_elements':self.n_elements,'learning_rate':self.learning_rate}
    
    #public function to train the ensemble
    def fit(self, X_train : np.array, y_train : np.array) -> None:
        #initialize sample weights, residuals, & model array
        w              = np.ones((y_train.shape[0]))
        self.residuals = []
        self.f         = []
        #loop through the specified number of iterations in the ensemble
        for _ in range(self.n_elements):
            #make a copy of the weak learner
            model = clone(self.weak_learner)
            #fit the weak learner on the current dataset
            model.fit(X_train,w)
            #update the sample weights
            y_pred = model.predict(X_train)
            m      = y_pred != y_train
            w[m]  *= np.exp(self.learning_rate)
            #append resulting model
            self.f.append(model)
            #append current count of correctly labeled samples
            self.model_weights.append(np.sum(y_pred == y_train))
            #append current f1 score
            if self.record_f1s:
                self.f1s.append(self.__compute_f1(X_train,y_train))
        #normalize the model weights
        self.model_weights /= np.sum(self.model_weights)
      
    # private function to compute f1 score for n-trained weak learners
    def __compute_f1(self, X_train : np.array, y_train : np.array) -> float:
        #initialize output
        y_pred = np.zeros((X_train.shape[0]))
        #normalize model weights
        n_model_weights = self.model_weights/np.sum(self.model_weights)
        #traverse ensemble to generate predictions
        for model,mw in zip(self.f,n_model_weights):
            y_pred += mw*model.predict(X_train)
        #combine output from ensemble
        y_pred = np.round(y_pred).astype(int)
        #return computed f1 score
        return(f1_score(y_train,y_pred))
    
    #public function to return training f1 scores
    def get_f1s(self) -> List:
        return(self.f1s)
    
    #public function to generate predictions
    def predict(self, X_test : np.array) -> np.array:
        #initialize output
        y_pred = np.zeros((X_test.shape[0]))
        #traverse ensemble to generate predictions
        for model,mw in zip(self.f,self.model_weights):
            y_pred += mw*model.predict(X_test)
        #combine output from ensemble
        y_pred = np.round(y_pred).astype(int)
        #return predictions
        return(y_pred)
