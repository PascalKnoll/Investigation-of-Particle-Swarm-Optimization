from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.base import BaseEstimator
import pyswarms as ps
import numpy as np

class GPR(BaseEstimator):
    def __init__(self, c1: float=0.5, c2: float=0.3, w: float=0.9, n_optim_steps: int=10, n_particles: int=10, n_restarts_optimizer: int=10) -> None:
        """
        :param c1: cognitive parameter
        :param c2: social parameter
        :param w: inertia weight
        :param n_optim_steps: number of optimization steps
        :param n_particles: number of particles
        :param n_restarts_optimizer: number of restarts of the optimizer
        """
        self.c1 = c1
        self.c2 = c2
        self.w = w

        self.options = {'c1': c1, 'c2': c2, 'w': w}

        self.n_optim_steps = n_optim_steps
        self.n_particles = n_particles
        self.n_restarts_optimizer = n_restarts_optimizer

        self.model = GaussianProcessRegressor(n_restarts_optimizer=n_restarts_optimizer, optimizer=self._optim)

    
    def hyper_optimize(self, X, y, grid=None):
        if grid is None:
            grid = {
                'c1': np.linspace(0.1, 1, 5),
                'c2': np.linspace(0.1, 1, 5),
                'w': np.linspace(0.1, 1, 5),
                'n_optim_steps': np.linspace(1, 10, 5),
                'n_particles': np.linspace(1, 10, 5),
                'n_restarts_optimizer': np.linspace(1, 10, 5)
            }
        
        clf = GridSearchCV(estimator=self, param_grid=grid, verbose=1, scoring=self._scoring, n_jobs=1, error_score='raise')
        X = X.T
        y = y[..., None]
        clf = clf.fit(X, y)
        self.model = clf.best_estimator_
        return self


    def fit(self, X: np.array, y: np.array) -> None:
        """
        :param X: training data
        :param y: training labels
        :return: self
        """
        if X.shape[0] < X.shape[1]: X = X.T
        if len(y.shape) == 1: y = y[..., None]
        self.model = self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        """
        :param X: test data
        :return: predictions
        """
        if X.shape[0] < X.shape[1]: X = X.T
        y = self.model.predict(X)
        return y


    def _optim(self, obj_func: callable, init_theta: np.array, bounds: np.array) -> tuple:
        """
        :param obj_func: objective function
        :param init_theta: initial theta
        :param bounds: bounds of theta
        :return: best theta
        """
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=len(init_theta), options=self.options, bounds=bounds)
        f_opt, theta_opt = optimizer.optimize(obj_func, iters=self.n_optim_steps, verbose=False)

        return theta_opt, f_opt
        
    def _scoring(self, estimator, X, y):
        y_pred = estimator.predict(X)
        return mean_squared_error(y, y_pred)
    

def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    return np.mean((y_true - y_pred) ** 2)