import numpy

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self, lam=0.0,penalize_constant = True):
        
        self.lam = lam
        self.penalize_constant = penalize_constant
            
    def fit(self, X, y):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        y : Array of shape [n_samples, 1]
        """        

        # make sure that we have multidimensional numpy arrays
        X = numpy.array(X).reshape((X.shape[0], -1))
        # IMPORTANT: Make sure that we have a column vector! 
        y = numpy.array(y).reshape((len(y), 1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)           

        # compute weights (solve system of linear equations)
        diag = self.lam * X.shape[0] * numpy.identity(X.shape[1])
        if not self.penalize_constant:
            diag[0, 0] = 0.0
        a = numpy.dot(X.T, X) + diag
        b = numpy.dot(X.T, y)
        self._w = numpy.linalg.solve(a,b)
                
    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     

        # make sure that we have multidimensional numpy arrays
        X = numpy.array(X).reshape((X.shape[0], -1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)           

        # compute predictions
        predictions = numpy.dot(X, self._w)

        return predictions
