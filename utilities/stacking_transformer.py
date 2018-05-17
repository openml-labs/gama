from sklearn.base import TransformerMixin
import numpy as np


def make_stacking_transformer(estimator_type):
    """Create a class that is a TransformerMixin for an arbitrary sklearn estimator
    by adding the predictions of the estimator trained on the data, to the data.
    Returns the class.
    """
    
    class StackingTransformer(estimator_type, TransformerMixin):
        """ Wrapper for a sklearn estimator to make it a transformer by
        appending predictions of the data to the dataset.
        """
    
        def transform(self, X):
            """ Transform the data by adding a column of predictions. """
            # It might be more pure to make the predictions with k-Fold CV.
            # In that case all the estimates on the test sets combined form
            # the new column. However, this is roughly k times as expensive.
            # Due to the nature of EA needing quick evaluations, we forgo this
            # for now.
            y_pred = self.predict(X)
            y_pred = y_pred.reshape(X.shape[0],1)
            return np.append(X, y_pred, axis=1)
        
    return StackingTransformer
