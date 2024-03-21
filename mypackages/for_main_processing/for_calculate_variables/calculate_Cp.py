# mypy: ignore-errors
import numpy as np
from numpy import sin, cos, deg2rad

from scipy.interpolate import interp1d
from pathlib import Path

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

from for_pre_processing.input_from_Setting import Constants
from for_main_processing.for_calculate_variables.calculate_aero_coefficients import Cn_alpha_roll



ALPHA_MAX = Constants.rocket()['Alpha_Max']
Cp_ALPHA_0 = Constants.rocket()['Cp']['Alpha_0']
Cp_ALPHA_MAX = Constants.rocket()['Cp']['Alpha_Max']


def Cp(alpha):

    Cp_interpolated = interp1d([0, ALPHA_MAX], [Cp_ALPHA_0, Cp_ALPHA_MAX], bounds_error=False, fill_value=(Cp_ALPHA_0, Cp_ALPHA_MAX))
    Cp = Cp_interpolated(np.abs(alpha))

    return Cp

def Cp_roll(alpha, roll):

    Cp_Cn_ALPHA_EXCEPT_r = Constants.rocket()['Cp_Cn_Alpha_Except_r']
    Cp_Cn_ALPHA_r_MAX = Constants.rocket()['Cp_Cn_Alpha_r_Max']

    return (Cp_Cn_ALPHA_EXCEPT_r + Cp_Cn_ALPHA_r_MAX * cos(deg2rad(roll))**2) / Cn_alpha_roll(roll)


if __name__ == '__main__':

    alpha = np.linspace(-15, 15, num=100)

    Cp = Cp(alpha)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(alpha, Cp, 'b', label='Cp [m]')

    ax.set_xlabel('Alpha [deg]')
    ax.grid()
    handler1, label1 = ax.get_legend_handles_labels()
    ax.legend(handler1, label1, loc='lower right')

    plt.show()