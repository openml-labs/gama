from sklearn.base import TransformerMixin
import numpy as np

def make_stacking_transformer(estimator_type):
    
    class StackingTransformer(estimator_type, TransformerMixin):
    
        def transform(self, X):
            y_pred = self.predict(X)
            y_pred = y_pred.reshape(X.shape[0],1)
            return np.append(X, y_pred, axis=1)
        
    return StackingTransformer