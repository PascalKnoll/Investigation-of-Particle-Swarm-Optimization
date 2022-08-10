# Investigation of Particle Swarm Optimisation for the Optimisation of Gaussian Process Models

*Repository for lecture "Applied Optimisation Techniques" exam at DHBW Ravensburg*

### by Timo Heiß, Tom Zehle, Nick Hillebrand and Pascal Knoll

The movements of swarms, whether of birds or fish, have always been a fascinating phenomenon of nature. They consist of individual animals and yet seem to move as a large whole.  The animals seem to be arranged randomly and yet the overall movement is precisely synchronised. And as if by some magical power, the swarm seems to be controlled by a central controlling authority.

The idea of Particle Swarm Optimisation (PSO) is based on the concept of swarms (cf. Kennedy and Eberhart, 1995). This optimisation algorithm is subject of the research of this project. Specifically, the optimisation problem of Maximum Likelihood Estimation (MLE) for Gaussian Process Models (GPM) is investigated. This problem is to be solved with the help of PSO, whereby the functioning and suitability of the optimisation algorithm is empirically investigated.

First, an in-depth, empirical investigation of the optimisation algorithm PSO is deducted regarding the training of a GPM on a fixed data set. In order to better assess the results, a comparison is made with other optimisation algorithms, *Differential Evolution* and *Random Search*. To confirm these results, further research is then carried out, in which various influencing variables are systematically varied. In doing so, the likelihood function changes, which allows to investigate different optimization problems. Lastly, a closer look at the hyperparameters of PSO is taken and their influence on the optimizer's performance is assesed.

The results of the research are described in the corresponding paper, which can be found in this repository as soon as it is finished.


### Literature:

[Kennedy and Eberhart 1995] Kennedy, J.; Eberhart, R.: Particle swarm optimization. In: Proceedings of ICNN’95 - International Conference on Neural Networks 4th edition, 1995, pp. 1942–1948 vol.4
