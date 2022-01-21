# dens_deriv_kernel_estim
Python implementations of kernel-based density derivative and density derivative ratio estimators. <br />
For density difference and density ratio estimators Python implementations exist, so those are not included here (also for copyright reasons).

This repository contains Python codes for the following methods:
* MISED = Mean Integrated Squared Error for Derivatives (density derivative estimator)
* LSDDR = Least Squares Density Derivative Ratio (density derivative ratio estimator)

The current implementations of MISED and LSDDR are either new or loosely based on existing MATLAB versions
 (see references below). The purpose of this repository is to allow the use of both methods with a
 similar call pattern and usage, as shown in the Jupyter notebook demo. 

References for the methods in this repo:
* MISED:
  * Method by Sasaki et. al. (2015) [https://doi.org/10.1162/NECO_a_00835]
* LSDDR:
  * Method by Sasaki et. al. (2017) [http://proceedings.mlr.press/v54/sasaki17a.html], 
  MATLAB code variant for Least-Squares Density Ridge Finder (LSDRF) by Hiroaki Sasaki 
  [https://sites.google.com/site/hworksites/home/software/lsdrf]

References for other estimators, where Python codes already exist:
* LSDD -- Least Squares Density Difference (density difference estimator):
  * Code by M.C. du Plessis (http://www.ms.k.u-tokyo.ac.jp/sugi/software.html#LSDD), 
  method by Sugiyama et. al. (2012) [https://doi.org/10.1162/NECO_a_00492]
* RuLSIF -- Relative unconstrained Least Squares Importance Fitting (relative density ratio estimator):
  * Code by Koji MAKIYAMA & Ameya Daigavane (https://github.com/hoxo-m/densratio_py), 
  method by Yamada et. al. (2012) [https://doi.org/10.1162/NECO_a_00442], 
  based on uLSIF method of Kanamori et. al. (2009) [https://www.jmlr.org/papers/volume10/kanamori09a/kanamori09a.pdf]

