# -*- coding: utf-8 -*-

from pprint import pformat
from re import sub


class DensityDerivative:
    """Density Derivative."""
    def __init__(self, method, theta, lambda_, kernel_info, compute_density_deriv):
        self.method = method
        self.theta = theta
        self.lambda_ = lambda_
        self.kernel_info = kernel_info
        self.compute_density_deriv = compute_density_deriv

    def __str__(self):
        return """
Method: %(method)s

Kernel Information:
%(kernel_info)s

Kernel Weights (theta):
  %(theta)s

Regularization Parameter (lambda): %(lambda_)s

Function to Estimate Density Derivative Ratio:
  compute_density_deriv_ratio(x)

"""[1:-1] % dict(method=self.method, kernel_info=self.kernel_info,
                 theta=my_format(self.theta), lambda_=my_format(self.lambda_))


class KernelInfo:
    """Kernel Information."""
    def __init__(self, kernel_type, kernel_num, sigma, centers):
        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        self.sigma = sigma
        self.centers = centers

    def __str__(self):
        return """
  Kernel type: %(kernel_type)s
  Number of kernels: %(kernel_num)s
  Bandwidth(sigma): %(sigma)s
  Centers: %(centers)s
"""[1:-1] % dict(kernel_type=self.kernel_type, kernel_num=self.kernel_num,
                 sigma=my_format(self.sigma), centers=my_format(self.centers))


def my_format(str):
    return sub(r"\s+", " ", (pformat(str).split("\n")[0] + ".."))
