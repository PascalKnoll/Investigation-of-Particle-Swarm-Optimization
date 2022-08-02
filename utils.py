from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.base import BaseEstimator
import pyswarms as ps
import numpy as np
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher
from IPython.display import Image
import matplotlib.pyplot as plt

from scipy.linalg import cho_solve
from numpy.linalg import cholesky
from tqdm import tqdm

from numba import njit, jit


GPR_CHOLESKY_LOWER = True

class GPR(BaseEstimator):
    def __init__(self, kernel=None, c1: float=0.5, c2: float=0.3, w: float=0.7, n_optim_steps: int=20, n_particles: int=10) -> None:
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
        self.kernel = kernel

        self.n_optim_steps = n_optim_steps
        self.n_particles = n_particles

        self.model = GaussianProcessRegressor(kernel=self.kernel, optimizer=self._optim, alpha=1e-3)
        theta_dim = len(kernel.theta)
        self.optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, 
            bounds=([-11.51292546]*theta_dim, [11.51292546]*theta_dim), 
            dimensions=theta_dim, options={'c1': c1, 'c2': c2, 'w': w})

    
    def hyper_optimize(self, X, y, grid=None):
        if grid is None:
            grid = {
                'c1': np.linspace(0.1, 1, 5),
                'c2': np.linspace(0.1, 1, 5),
                'w': np.linspace(0.1, 1, 5),
                'n_particles': np.linspace(1, 10, 5)
            }
        
        clf = GridSearchCV(estimator=self, param_grid=grid, verbose=1, scoring=self._scoring, n_jobs=1, error_score='raise')
        X = X.T
        y = y[..., None]
        clf = clf.fit(X, y)
        self.model = clf.best_estimator_
        return self


    def fit(self, X: np.array, y: np.array):
        """
        :param X: training data
        :param y: training labels
        :return: self
        """
        if X.shape[0] < X.shape[1]: X = X.T
        if len(y.shape) == 1: y = y[..., None]
        self.model = self.model.fit(X, y)

        return self


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
        :return: best theta, best cost
        """
        f_opt, theta_opt = self.optimizer.optimize(
            lambda thetas: -np.array([self.model.log_marginal_likelihood(theta) for theta in thetas]), 
            iters=self.n_optim_steps
            )
        return theta_opt, f_opt

    # @njit
    def asdf(self, thetas):
        n_batches = thetas.shape[0]
        errors = np.zeros(thetas.shape[0], dtype=np.float64)
        for i in tqdm(range(n_batches)):
            errors[i] = self.obj_func(thetas[i])
        return errors
        # return self.obj_func2(thetas)

    # @njit
    def obj_func(self, theta):
        kernel = self.model.kernel_
        kernel.theta = theta
        
        K = kernel(self.model.X_train_)

        K[np.diag_indices_from(K)] += self.model.alpha
        L = cholesky(K)

        y_train = self.model.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        
        log_likelihood = log_likelihood_dims.sum()

        return log_likelihood     

        




    def _scoring(self, estimator, X, y):
        y_pred = estimator.predict(X)
        return mean_squared_error(y, y_pred)


    def visualize_training(self, obj_function, optimizer):
        m = Mesher(func=obj_function)
        # Make animation
        animation = plot_contour(pos_history=optimizer.pos_history,
                        mesher=m,
                        mark=(0,0))
        animation.save('mymovie.mp4')


    def plot_history(optimizer):
        plot_cost_history(cost_history=optimizer.cost_history)
        plt.show()

        

def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def visualize(X, y, title=None):
    plt.scatter(X[0], X[1], c=y)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


def visualize_meshgrid(x, y, target_func=None, title=None):
    Xs = np.array(np.meshgrid(x, y))
    z = np.asarray([target_func(X) for X in Xs])
    plt.pcolormesh(x, y, z)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


def generate_sample(n, n_dims, lower, upper, target_func, noise_scale=0):
    """
    Generates data sample.
    """
    rs = np.random.RandomState(42)
    X = rs.rand(n_dims,n)
    for i in range(n_dims):
        X[i] = X[i]*(upper[i]-lower[i]) + lower[i]
    y = target_func(X)
    y += rs.normal(0, noise_scale, size=y.shape)
    
    return (X,y)



def aha(thetas, model):
    n_batches = thetas.shape[0]
    kernel = model.kernel_
    X_train = model.X_train_
    n_train = X_train.shape[0]
    y_train = model.y_train_
    alpha = model.alpha
    Ks = np.zeros((n_batches, n_train, n_train), dtype=np.float64)
    for i in range(n_batches):
        kernel.theta = thetas[i]
        Ks[i] = kernel(X_train)

    return asdf(thetas, Ks, y_train, alpha)

@jit
def asdf(thetas, Ks, y_train, alpha):
    n_batches = thetas.shape[0]
    errors = np.zeros(thetas.shape[0], dtype=np.float64)   
    for i in range(n_batches):
        errors[i] = obj_func(Ks[i], y_train, alpha)
    return errors

@jit
def obj_func(K, y_train, alpha):
    K[np.diag_indices_from(K)] += alpha
    L = cholesky(K)

    # alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

    log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
    log_likelihood_dims -= np.log(np.diag(L)).sum()
    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    
    log_likelihood = log_likelihood_dims.sum()

    return log_likelihood

from scipy.spatial.distance import pdist, cdist, squareform
# import kernels
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter
class HHKZ(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(
        self, 
        length_scale=1.0, 
        length_scale_bounds=(1e-5, 1e5),
        # sigma=1.0,
        # sigma_bounds=(1e-5, 1e5)
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        # self.sigma = sigma
        # self.sigma_bounds = sigma_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)
    
    # @property
    # def hyperparameter_sigma(self):
    #     if self.anisotropic:
    #         return Hyperparameter(
    #             "sigma",
    #             "numeric",
    #             self.sigma_bounds,
    #             len(self.sigma),
    #         )
    #     return Hyperparameter("sigma", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            dists = pdist(X * self.length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        return K
