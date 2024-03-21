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




ALPHA_MAX = Constants.rocket()['Alpha_Max']

def Cn(alpha):
    Cn_ALPHA = Constants.rocket()['Cn_Alpha']

    Cn_interpolated = interp1d([0, ALPHA_MAX], [0, Cn_ALPHA * deg2rad(ALPHA_MAX)], bounds_error=False, fill_value=(0, Cn_ALPHA * deg2rad(ALPHA_MAX)))
    Cn = Cn_interpolated(np.abs(alpha))

    return Cn


def Cl(alpha):
    Cl_ALPHA = Constants.rocket()['Cl_Alpha']

    Cl_interpolated = interp1d([0, ALPHA_MAX], [0, Cl_ALPHA * deg2rad(ALPHA_MAX)], bounds_error=False, fill_value=(0, Cl_ALPHA * deg2rad(ALPHA_MAX)))
    Cl = Cl_interpolated(np.abs(alpha))

    return Cl


Cd0 = Constants.rocket()['Cd0']
Cd_Cl = Constants.rocket()['Coefficient_Cd_Induced']

Ct0 = Constants.rocket()['Ct0']
Ct_Cn = Constants.rocket()['Coefficient_Ct_Induced']

def Cd(alpha):

    Cd = Cd0 + Cd_Cl * Cl(alpha)**2

    return Cd

def Ct(alpha):

    Ct = Ct0 + Ct_Cn * Cn(alpha)**2

    return Ct


def Cn_alpha_roll(roll):

    Cn_ALPHA_EXCEPT_r = Constants.rocket()['Cn_Alpha_Except_r']
    Cn_ALPHA_r_MAX = Constants.rocket()['Cn_Alpha_r_Max']

    return Cn_ALPHA_EXCEPT_r + Cn_ALPHA_r_MAX * cos(deg2rad(roll))**2


def Cn_roll(alpha, roll):

    alpha = np.abs(alpha)
    if isinstance(alpha, np.ndarray):
        alpha[alpha > ALPHA_MAX] = ALPHA_MAX

    else:
        if alpha > ALPHA_MAX:
            alpha = ALPHA_MAX

    return Cn_alpha_roll(roll) * deg2rad(alpha)

def Ct_roll(alpha, roll):

    return Ct0 + Ct_Cn * Cn_roll(alpha, roll)**2

def Cl_roll(alpha, roll):

    alpha = np.abs(alpha)
    if isinstance(alpha, np.ndarray):
        alpha[alpha > ALPHA_MAX] = ALPHA_MAX

    else:
        if alpha > ALPHA_MAX:
            alpha = ALPHA_MAX

    return Cn_roll(alpha, roll) * cos(deg2rad(alpha)) - Ct_roll(alpha, roll) * sin(deg2rad(alpha))

def Cd_roll(alpha, roll):

    return Cd0 + Cd_Cl * Cl_roll(alpha, roll)**2


DELTA_MAX = Constants.rocket()['Delta_Max']
Cmt_DELTA = Constants.rocket()['Cmt_Delta']

def Cmt_roll_control(delta):

    Cmt_roll_control_interpolated = interp1d([0, DELTA_MAX], [0, Cmt_DELTA * deg2rad(DELTA_MAX)], bounds_error=False, fill_value=(0, Cmt_DELTA * deg2rad(DELTA_MAX)))
    Cmt_roll_control = Cmt_roll_control_interpolated(np.abs(delta))

    return Cmt_roll_control



if __name__ == '__main__':

    alpha = np.linspace(-15, 15, num=100)

    var_Cl = Cl_roll(alpha, 0)
    var_Cn = Cn_roll(alpha, 0)
    var_Cd = Cd_roll(alpha, 0)
    var_Ct = Ct_roll(alpha, 0)

    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0, 0].plot(alpha, var_Cl, 'r', label='Cl')

    axes[0, 0].set_xlabel('Alpha [deg]')
    axes[0, 0].grid()
    handler1, label1 = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handler1, label1, loc='lower right')


    axes[0, 1].plot(alpha, var_Cn, 'g', label='Cn')

    axes[0, 1].set_xlabel('Alpha [deg]')
    axes[0, 1].grid()
    handler1, label1 = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handler1, label1, loc='lower right')


    axes[1, 0].plot(alpha, var_Cd, 'b', label='Cd')

    axes[1, 0].set_xlabel('Alpha [deg]')
    axes[1, 0].grid()
    handler1, label1 = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handler1, label1, loc='lower right')


    axes[1, 1].plot(alpha, var_Ct, 'gold', label='Ct')

    axes[1, 1].set_xlabel('Alpha [deg]')
    axes[1, 1].grid()
    handler1, label1 = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handler1, label1, loc='lower right')


    plt.show()