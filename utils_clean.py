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


def visualize_pred_meshgrid(predictor, title):
    m = np.arange(-2.5,1.5,0.01)
    p = np.arange(-1.5,2.5,0.01)
    X = np.array(np.meshgrid(m, p))
    Y = np.zeros((400,400))
    for i in range(len(X.T)):
        Y[i] = predictor.predict(X.T[i]).flatten()
    plt.pcolormesh(np.array(np.meshgrid(m, p))[0], np.array(np.meshgrid(m, p))[1], Y, cmap="inferno")
    plt.ylabel("$x_2$")
    plt.xlabel("$x_1$")
    plt.colorbar()
    plt.title(title)


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

def get_likelihood_grids(gpr, bounds=(-11.5, 11.5), n=100):
        # visualize the likelihood for each point in the grid, given by the function fun
        x_grid = np.linspace(bounds[0], bounds[1], n)
        y_grid = x_grid
        X_grid = np.array(np.meshgrid(x_grid, y_grid))
        X_grid  = X_grid.swapaxes(0, 2)
        X_grid = X_grid.reshape((-1, 2))
        z = np.asarray([gpr.log_marginal_likelihood(X) for X in tqdm(X_grid)])
        z = z.reshape(n, n)

        return x_grid, y_grid, z

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

class Optimizer:
    def __init__(self):
        self.pos_hist = []

    def optim(self, obj_func, init_theta, bounds):
        raise NotImplementedError
    
    def visualize_optimization(self, x_grid, y_grid, z, file_path="blockbuster.mp4", marker="*", show_zaeff=True):
        def animate(i):
            ax = plt.axes()
            plt.cla()
            # plt.clf()
            ax.pcolormesh(x_grid, y_grid, z, cmap="inferno")
            plt.xlabel(r"$\theta_1$", fontsize=14)
            plt.ylabel(r"$\theta_2$", fontsize=14)
            plt.text(s=f"Step {i}", x=-5, y=-5)
            plt.title("Likelihood", y=1.1, fontsize=18)
            for pos in self.pos_hist[i]:
                if show_zaeff:
                    imagebox = OffsetImage(im_arr, zoom=0.15)
                    ab = AnnotationBbox(imagebox, (pos[0], pos[1]), frameon=False)
                    ax.add_artist(ab)
                else:
                    ax.scatter(pos[0], pos[1], c='red', marker=marker)
            return ax,
        if self.pos_hist == []:
            return
        im_arr = plt.imread("zaefferer.png")
        fig = plt.figure(dpi=600)
        anim = FuncAnimation(fig, animate, frames=len(self.pos_hist), interval=500)
        anim.save(file_path)


class DEOptim(Optimizer):
    def __init__(self, visualize=False, maxiter=10, popsize=10):
        super().__init__()
        self.optimize = self.optimize_with_visulization if visualize else self.optimize
        self.maxiter = maxiter
        self.popsize = popsize

    def optimize(self, obj_func, init_theta, bounds):
        res = differential_evolution(lambda x: obj_func(x)[0], bounds, x0=None, maxiter=self.maxiter, popsize=self.popsize)
        return res.x, res.fun

    def optimize_with_visulization(self, obj_func, init_theta, bounds):
        finished = False
        current_theta = init_theta
        
        while not finished:
            res = differential_evolution(lambda x: obj_func(x)[0], bounds, x0=current_theta, maxiter=1)
            current_theta = res.x
            self.pos_hist.append([current_theta])
            finished = res.success

        return res.x, res.fun


class RandomOptim(Optimizer):
    def __init__(self, maxiter, visualize=False, random_state=42):
        super().__init__()
        self.maxiter = maxiter
        self.visualize = visualize
        self.random_state = random_state

    def optimize(self, obj_func, init_theta, bounds):
        # optimal thetas
        theta_opt = []
        # optimal log likelihood, starts with a very bad value
        func_max = float("inf")
        # current log likelihood
        func_current = 0
        # current thetas
        thetas = []
        rs = np.random.RandomState(self.random_state)
        for _ in range(0, self.maxiter):
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
    def __init__(self, c1=0.5, c2=0.3, w=0.9, n_particles=10, n_iters=10, init_pos=None, visualize=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.visualize = visualize
        self.init_pos = init_pos
        
    def optimize(self, obj_func, init_theta, bounds):
        theta_dim = len(init_theta)
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, 
            bounds=([-11]*theta_dim, [11]*theta_dim),
            dimensions=theta_dim, 
            init_pos=self.init_pos,
            options={'c1': self.c1, 'c2': self.c2, 'w': self.w}
        )

        f_opt, theta_opt = optimizer.optimize(
            lambda thetas: [obj_func(theta)[0] for theta in thetas], 
            iters=self.n_iters,
            verbose=False
        )

        if self.visualize:
            self.pos_hist = optimizer.pos_history

        return theta_opt, f_opt