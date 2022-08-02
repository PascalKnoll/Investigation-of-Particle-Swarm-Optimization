from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.base import BaseEstimator
import pyswarms as ps
import numpy as np
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher
from IPython.display import Image
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from scipy.linalg import cho_solve
from numpy.linalg import cholesky
from tqdm import tqdm

from matplotlib.animation import FuncAnimation 


def visualize_meshgrid(x, y, target_func, title=None):
    X = np.array(np.meshgrid(x, y))
    Z = target_func(X)
    plt.pcolormesh(x,y,Z, cmap="inferno")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


def generate_sample(n, n_dims, lower, upper, target_func, noise_scale=0, random_state=42):
    """
    Generates data sample.
    """
    rs = np.random.RandomState(random_state)
    X = rs.rand(n_dims,n)
    for i in range(n_dims):
        X[i] = X[i]*(upper[i]-lower[i]) + lower[i]
    y = target_func(X)
    y += rs.normal(0, noise_scale, size=y.shape)
    
    return (X,y)

def get_log_likelihood(gpr, bounds=(-10, 10), n=10,):
        # visualize the likelihood for each point in the grid, given by the function fun
        x_grid = np.linspace(bounds[0], bounds[1], n)
        y_grid = x_grid
        X_grid = np.array(np.meshgrid(x_grid, y_grid))
        X_grid  = X_grid.swapaxes(0, 2)
        X_grid = X_grid.reshape((-1, 2))
        z = np.asarray([gpr.log_marginal_likelihood(X) for X in tqdm(X_grid)])
        z = z.reshape(n, n)

        return x_grid, y_grid, z

class Optimizer:
    def __init__(self):
        self.pos_hist = []

    def optim(self, obj_func, init_theta, bounds):
        raise NotImplementedError
    
    def visualize_optimization(self, x_grid, y_grid, z, file_path="blockbuster.mp4"):
        def animate(i):
            ax = plt.axes()
            ax.pcolormesh(x_grid, y_grid, z)
            for pos in self.pos_hist[i]:
                ax.scatter(pos[0], pos[1], c='gray')
            return ax,

        anim = FuncAnimation(plt.figure(), animate, frames=len(self.pos_hist), interval=500)
        anim.save(file_path)

class DEOptim(Optimizer):
    def __init__(self, visualize=False):
        super().__init__()
        self.visualize = visualize

    def optimize(self, func, init_theta, bounds):
        res = differential_evolution(lambda x: func(x)[0], bounds, x0=init_theta)
        return res.x, res.fun

class RandomOptim(Optimizer):
    def __init__(self, n_iters, visualize=False):
        super().__init__()
        self.n_iters = n_iters
        self.visualize = visualize

    def optimize(self, obj_func, init_theta, bounds):
        # optimal thetas
        theta_opt = []
        # optimal log likelihood, starts with a very bad value
        func_max = float("inf")
        # current log likelihood
        func_current = 0
        # current thetas
        thetas = []
        rs = np.random.RandomState(42)
        for _ in range(0, self.n_iters):
            thetas = []
            for _ in range(0, init_theta.shape[0]):
                thetas.append(rs.uniform(bounds[0][0],bounds[0][1]))

            func_current = obj_func(thetas)[0]

            if func_current < func_max:
                func_max = func_current
                theta_opt = thetas

            if self.visualize:
                self.pos_hist.append(theta_opt)

        if self.visualize:
            self.pos_hist = np.asarray(self.pos_hist).reshape(-1, 1, 2)
        return theta_opt, func_max


class PSOOptim(Optimizer):
    def __init__(self, c1, c2, w, n_particles, visualize=False):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.n_particles = n_particles

    def optimize(self, obj_func, init_theta, bounds):
        theta_dim = len(init_theta)
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, 
            bounds=(np.asarray(bounds).T), 
            dimensions=theta_dim, options={'c1': self.c1, 'c2': self.c2, 'w': self.w})

        f_opt, theta_opt = optimizer.optimize(lambda thetas: [obj_func(theta) for theta in thetas])
        if self.visualize:
            self.pos_hist = optimizer.pos_history
        return theta_opt, f_opt