from sklearn.base import TransformerMixin
import numpy as np

def make_stacking_transformer(estimator_type):
    
    class StackingTransformer(estimator_type, TransformerMixin):
    
        def transform(self, X):
            y_pred = self.predict(X)
            y_pred = y_pred.reshape(X.shape[0],1)
            return np.append(X, y_pred, axis=1)
        
    return StackingTransformer

"""
class StackingTransformer(BaseEstimator, TransformerMixin):
    
    def transform(self, X, y):
        self.fit(X, y)
        y_pred = self.predict(X)
        y_pred = y.reshape(len(y),1)
        return np.append(X, y_pred, axis=1)
    
class DTStackingTransformer(DecisionTreeClassifier, TransformerMixin):
    
    def transform(self, X):
        y_pred = self.predict(X)
        y_pred =y_pred.reshape(X.shape[0],1)
        # if has predict proba then also add that
        return np.append(X, y_pred, axis=1)
    
class StackingTransformer(TransformerMixin):
    
    def init(self, estimator):
        self.__dict__
    
    def transform(self, X):
        y_pred = self.predict(X)
        y_pred =y_pred.reshape(X.shape[0],1)
        # if has predict proba then also add that
        return np.append(X, y_pred, axis=1)
"""