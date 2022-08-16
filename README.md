# Investigation of Particle Swarm Optimisation for the Optimisation of Gaussian Process Models

*Repository for lecture ["Applied Optimisation Techniques" exam at DHBW Ravensburg](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/Projektbeschreibung.Rmd)*

### by [Timo Heiß](https://www.linkedin.com/in/timo-hei%C3%9F/), [Tom Zehle](https://www.linkedin.com/in/tom-zehle/), [Nick Hillebrand](https://www.linkedin.com/in/nick-hillebrand-395466218/) and [Pascal Knoll](https://www.linkedin.com/in/knoll-pascal/)

The movements of swarms, whether of birds or fish, have always been a fascinating phenomenon of nature. They consist of individual animals and yet seem to move as a large whole.  The animals seem to be arranged randomly and yet the overall movement is precisely synchronised. And as if by some magical power, the swarm seems to be controlled by a central controlling authority.

The idea of Particle Swarm Optimisation (PSO) is based on the concept of swarms (cf. Kennedy and Eberhart, 1995). This optimisation algorithm is subject of the research of this project. Specifically, the optimisation problem of Maximum Likelihood Estimation (MLE) for Gaussian Process Models (GPM) is investigated. This problem is to be solved with the help of PSO, whereby the functioning and suitability of the optimisation algorithm is empirically investigated.

First, an in-depth, empirical investigation of the optimisation algorithm PSO is deducted regarding the training of a GPM on a fixed data set. In order to better assess the results, a comparison is made with other optimisation algorithms, *Differential Evolution* and *Random Search*. To confirm these results, further research is then carried out, in which various influencing variables are systematically varied. In doing so, the likelihood function changes, which allows to investigate different optimization problems. Lastly, a closer look at the hyperparameters of PSO is taken and their influence on the optimizer's performance is assesed.

The results of the research are described and evaluated in the corresponding paper, which can also be found in this repository. The outline of this paper is shown below. In case there is a corresponding notebook for a chapter on this repository, the link will directly lead to this notebook.

![plot](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/Plots/likelihood_landscape.png)
---

### Table of contents

 **[1. Introduction](#heading--1)**

**[2. Theoretical Foundations](#heading--2)**

  * [2.1. Gaussian Process Model and Maximum Likelihood Estimation](#heading--2-1)
  * [2.2. Particle Swarm Optimization](#heading--2-2)
  
**[3. Empirical Research](#heading--3)**

  * [3.1. Implementation](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization)
  * [3.2. Comparison to other Optimization Algorithms](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/AOT_Notebook_ch320.ipynb)
  * [3.3. Expanded Research](#heading--3-3)
    * [3.3.1. Variation in Number of Samples](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/AOT_Notebook_ch331.ipynb)
    * [3.3.2. Variation in Number of Dimensions](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/AOT_Notebook_ch332.ipynb)
    * [3.3.3. Variation of Groundtruth](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/AOT_Notebook_ch333.ipynb)
    * [3.3.4. Influence of Noise](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/AOT_Notebook_ch334.ipynb)
    * [3.3.5. Overall Evaluation](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/AOT_Notebook_ch335.ipynb)
  * [3.4. Hyperparameter Optimization for PSO](https://github.com/PascalKnoll/Investigation-of-Particle-Swarm-Optimization/blob/main/AOT_Notebook_ch340.ipynb)
  
**[4. Conclusion](#heading--4)**

---

### Literature:

[Kennedy and Eberhart 1995] Kennedy, J.; Eberhart, R.: Particle swarm optimization. In: Proceedings of ICNN’95 - International Conference on Neural Networks 4th edition, 1995, pp. 1942–1948 vol.4
