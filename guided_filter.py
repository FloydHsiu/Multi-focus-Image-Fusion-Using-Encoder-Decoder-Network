import numpy as np
import cv2 as cv
import tensorflow as tf


def guided_filter_cv(I, p, r, eps):
    h, w = I.shape
    r = (2*r, 2*r)

    mean_I = cv.boxFilter(I, -1, r)
    mean_p = cv.boxFilter(p, -1, r)
    mean_Ip = cv.boxFilter(I*p, -1, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    # this is the covariance of(I, p) in each local patch.

    mean_II = cv.boxFilter(I*I, -1, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    # Eqn. (5) in the paper
    b = mean_p - a * mean_I
    # Eqn. (6) in the paper

    mean_a = cv.boxFilter(a, -1, r)
    mean_b = cv.boxFilter(b, -1, r)

    q = mean_a * I + mean_b
    # Eqn. (8) in the paper
    return q
