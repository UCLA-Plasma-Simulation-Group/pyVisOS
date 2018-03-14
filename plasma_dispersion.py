import numpy as np
import scipy
from scipy import optimize


def zfunction(w):
    a = scipy.special.wofz(w)
    a *= np.sqrt(np.pi) * 1j
    return a


def zfunction_prime(w):
    return -2.0 * (1.0 + w * zfunction(w))


def zfunction_2prime(w):
    return -2.0 * (zfunction(w) + w * zfunction_prime(w))


def plasma_wave_w(wp, vth, k, maxwellian_convention_factor=2.0, inital_root_guess=None):
    chi_e = np.power((wp / (vth * k)), 2) / maxwellian_convention_factor

    def plasma_epsilon1(x):
        val = 1.0 - chi_e * zfunction_prime(x)
        return val

    if inital_root_guess is None:
        #    # use the Bohm-Gross dispersion formulas to get an initial guess for w
        inital_root_guess = np.sqrt(wp * wp + 3 * k * k * vth * vth)
        epsilon_root = scipy.optimize.newton(plasma_epsilon1, inital_root_guess)

    return epsilon_root * k * vth * np.sqrt(maxwellian_convention_factor)


def plasma_wave_vg(wp, vth, k, maxwellian_convention_factor=2.0, w=None):
    if w is None:
        w = plasma_wave_w(wp, vth, k, maxwellian_convention_factor)
    return np.real(2 * zfunction_prime(w) / (zfunction_2prime(w) * k))


def light_wave_vg(wp, k):
    return 1. / np.sqrt(wp * wp / (k * k) + 1.)
