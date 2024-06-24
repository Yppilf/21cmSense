import numpy as np
from matplotlib.pyplot import figure,colorbar
from astropy import units as un
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum, hera

def documentation():
    # print(help(PowerSpectrum))
    # print(help(Observation))
    # print(help(Observatory))
    # print(help(hera))
    print(help(Observatory.from_profile))

documentation()