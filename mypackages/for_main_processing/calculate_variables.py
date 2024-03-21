# mypy: ignore-errors
import numpy as np
from numpy import  rad2deg, power
from numpy.linalg import norm
from pathlib import Path


if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from for_pre_processing.input_from_Setting import Constants
    from for_main_processing import for_calculate_variables as FCV

else:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from for_pre_processing.input_from_Setting import Constants
    from . import for_calculate_variables as FCV


def thrust(time):

    thrust = FCV.Thrust.thrust(time)

    return thrust

def normalized_impulse(time):

    normalized_impulse = FCV.Thrust.normalized_impulse(time)

    return normalized_impulse

def linear_change(time):

    linear_change = FCV.Thrust.linear_change(time)

    return linear_change


def mass(time):

    mass_before = Constants.rocket()['Mass']['Before']
    mass_after = Constants.rocket()['Mass']['After']

    mass = (1 - normalized_impulse(time)) * mass_before + normalized_impulse(time) * mass_after

    return mass

def inertia(time):

    inertia_before = np.array(Constants.rocket()['Inertia']['Before'])
    inertia_after = np.array(Constants.rocket()['Inertia']['After'])

    if isinstance(time, np.ndarray):
        inertia = np.reshape((1 - normalized_impulse(time)), (-1, 1, 1)) * inertia_before + np.reshape(normalized_impulse(time), (-1, 1, 1)) * inertia_after

    else:
        inertia = (1 - normalized_impulse(time)) * inertia_before + normalized_impulse(time) * inertia_after

    return inertia

def Cg(time):

    Cg_before = Constants.rocket()['Cg']['Before']
    Cg_after = Constants.rocket()['Cg']['After']

    Cg = (1 - normalized_impulse(time)) * Cg_before + normalized_impulse(time) * Cg_after

    return Cg


def Cp(alpha, roll):

    return FCV.Cp_roll(alpha, roll)

def Fst(time, alpha, roll):

    L = Constants.rocket()['Body_Length']

    Fst = 100 * (Cg(time) - Cp(alpha, roll)) / L

    return Fst


def Cl(alpha, roll):

    return FCV.Aero.Cl_roll(alpha, roll)

def Cn(alpha, roll):

    return FCV.Aero.Cn_roll(alpha, roll)

def Cd(alpha, roll):

    return FCV.Aero.Cd_roll(alpha, roll)

def Ct(alpha, roll):

    return FCV.Aero.Ct_roll(alpha, roll)


def Cmt_roll_control(delta):

    return FCV.Aero.Cmt_roll_control(delta)


def air_density(altitude):

    R = Constants.physics()['R']
    P_STD = Constants.physics()['Pressure_STD']
    T_STD = Constants.physics()['Temperature_STD']

    air_density = (288.15**(-5.256) * P_STD / R) * power((T_STD + 273.15 - 6.5 * ((altitude + np.abs(altitude)) / (2 * 1000))), 4.256)

    return air_density

def dp(altitude, airflow):

    if airflow.ndim > 1:
        dp = (1/2) * air_density(altitude) * norm(airflow, axis=1)**2

    else:
        dp = (1/2) * air_density(altitude) * norm(airflow)**2

    return dp

def dp_as_kPa(altitude, airflow):

    return dp(altitude, airflow) / 1000

def alpha_airflow(airflow):

    if airflow.ndim > 1:
        alpha_airflow = np.arccos(-airflow[:, 0] / (norm(airflow, axis=1) + 1e-6))

    else:
        alpha_airflow = np.arccos(-airflow[0] / (norm(airflow) + 1e-6))

    return rad2deg(alpha_airflow)

def alpha(airflow):

    if airflow.ndim > 1:
        alpha = np.arctan(-airflow[:, 2] / (-airflow[:, 0] + 1e-6))

    else:
        alpha = np.arctan(-airflow[2] / (-airflow[0] + 1e-6))

    return rad2deg(alpha)

def beta(airflow):

    if airflow.ndim > 1:
        beta = np.arcsin(-airflow[:, 1] / (norm(airflow, axis=1) + 1e-6))

    else:
        beta = np.arcsin(-airflow[1] / (norm(airflow) + 1e-6))

    return rad2deg(beta)

def gamma(velocity_Cg):

    if velocity_Cg.ndim > 1:
        gamma = np.arctan(-velocity_Cg[:, 2] / (np.sqrt(velocity_Cg[:, 0]**2 + velocity_Cg[:, 1]**2) + 1e-6))

    else:
        gamma = np.arctan(-velocity_Cg[2] / (np.sqrt(velocity_Cg[0]**2 + velocity_Cg[1]**2) + 1e-6))

    return rad2deg(gamma)

def xi(velocity_Cg):

    if velocity_Cg.ndim > 1:
        xi = np.arctan(velocity_Cg[:, 1] / (velocity_Cg[:, 0] + 1e-6))

    else:
        xi = np.arctan(velocity_Cg[1] / (velocity_Cg[0] + 1e-6))

    return rad2deg(xi)


ROLL_TARGET = Constants.control()['Roll_Target']
TIME_START_CONTROL = Constants.control()['Time_Start_Control']
def roll_target(time):

    roll_target = ROLL_TARGET * (1 - (1 + (time - TIME_START_CONTROL)) * np.exp(-(time - TIME_START_CONTROL)))
    if isinstance(time, np.ndarray):
        roll_target[time < TIME_START_CONTROL] = 0

    else:
        if time < TIME_START_CONTROL:
            roll_target = 0

    return roll_target

def D_roll_target(time):

    D_roll_target = ROLL_TARGET * (time - TIME_START_CONTROL) * np.exp(-(time - TIME_START_CONTROL))
    if isinstance(time, np.ndarray):
        D_roll_target[time < TIME_START_CONTROL] = 0

    else:
        if time < TIME_START_CONTROL:
            D_roll_target = 0

    return D_roll_target

def e(time, roll):
    return roll_target(time) - roll

def De(time, D_roll):
    return D_roll_target(time) - D_roll


def delta_roll_control(Ie, e, De):

    K_p = Constants.control()['Gain']['K_p']
    K_i = Constants.control()['Gain']['K_i']
    K_d = Constants.control()['Gain']['K_d']

    return K_i * Ie + K_p * e + K_d * De

class Servo:

    def __init__(self):
        self.FREQUENCY = 2.0 #Hz
        self.PERIOD = 1 / self.FREQUENCY #s
        self.MAX_OUTPUT = 10.0 #deg
        self.MIN_OUTPUT = 2.0 #deg

        self.time_move = 7.01 #s
        self.output = 0.0 #deg

    def act(self, time, output):
        if isinstance(time, np.ndarray):
            time_ref_array = np.arange(time[0], time[-1], self.PERIOD)

            index_ref_array = np.array([np.where(np.isclose(time, time_ref))][0][0] for time_ref in time_ref_array)
            output_ref_array = np.array([output[index] for index in index_ref_array])

            index_ref_array = np.append(index_ref_array, len(time))

            for i in range(len(output_ref_array)):
                output[index_ref_array[i] : index_ref_array[i+1]] = output_ref_array[i]

            return output

        else:
            if time >= self.time_move:
                self.time_move += self.PERIOD
                self.output = output

            return self.output

    def delta_roll_control(self, Ie, e, De):

        K_p = Constants.control()['Gain']['K_p']
        K_i = Constants.control()['Gain']['K_i']
        K_d = Constants.control()['Gain']['K_d']

        delta = K_i * Ie + K_p * e + K_d * De
        delta = np.round(delta)

        if isinstance(delta, np.ndarray):
            delta[np.abs(delta) >= self.MAX_OUTPUT] = np.sign(delta[np.abs(delta) >= self.MAX_OUTPUT]) * self.MAX_OUTPUT

        else:
            if np.abs(delta) >= self.MAX_OUTPUT:
                delta = np.sign(delta) * self.MAX_OUTPUT

        return delta


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #print(inertia(time))
    #print(inertia(1.0))


    print(dp_as_kPa(1000, np.array([[0.0, 0.0, 0.0], [0, 0, 0]])))
    print(alpha_airflow(np.array([0, 0, 100])))
    print(xi(np.array([[100, 100, 0], [100, 0, 100]])))

    alt = np.linspace(-1000, 10000, num=100)
    var_alpha = np.linspace(-15, 15, num=100)
    time = np.linspace(0, 20, num=100)

    var_air_density = air_density(alt)
    var_Cg = Cg(time)
    var_mass = mass(time)
    var_Ixx, var_Iyy, var_Izz = inertia(time)[:, 0, 0], inertia(time)[:, 0, 0], inertia(time)[:, 2, 2]

    var_Cl = Cl(var_alpha, 0)
    var_Cn = Cn(var_alpha, 0)
    var_Cd = Cd(var_alpha, 0)
    var_Ct = Ct(var_alpha, 0)

    fig, axes = plt.subplots(nrows=2, ncols=4)

    axes[0, 0].plot(alt, var_air_density, 'b', label='rho [kg/m^3]')

    axes[0, 0].set_xlabel('Altitude [m]')
    axes[0, 0].grid()
    handler1, label1 = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handler1, label1, loc='upper right')


    axes[0, 1].plot(time, var_Cg, 'r', label='Cg [m]')

    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].grid()
    handler1, label1 = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handler1, label1, loc='lower right')


    axes[1, 0].plot(time, var_mass, 'black', label='mass [kg]')

    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].grid()
    handler1, label1 = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handler1, label1, loc='upper right')


    axes[1, 1].plot(time, var_Ixx, 'r', label='Ixx [kg*m^2]')
    axes[1, 1].plot(time, var_Iyy, 'b', label='Iyy [kg*m^2]')
    axes[1, 1].plot(time, var_Izz, 'g', label='Izz [kg*m^2]')

    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].grid()
    handler1, label1 = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handler1, label1, loc='lower right')


    axes[0, 2].plot(var_alpha, var_Cl, 'r', label='Cl')
    axes[0, 2].plot(var_alpha, var_Cn, 'g', label='Cn')

    axes[0, 2].set_xlabel('Alpha [deg]')
    axes[0, 2].grid()
    handler1, label1 = axes[0, 2].get_legend_handles_labels()
    axes[0, 2].legend(handler1, label1, loc='lower right')


    axes[1, 2].plot(var_alpha, var_Cd, 'b', label='Cd')
    axes[1, 2].plot(var_alpha, var_Ct, 'gold', label='Ct')

    axes[1, 2].set_xlabel('Alpha [deg]')
    axes[1, 2].grid()
    handler1, label1 = axes[1, 2].get_legend_handles_labels()
    axes[1, 2].legend(handler1, label1, loc='lower right')


    axes[0, 3].plot(time, roll_target(time), 'r', label='roll_target [deg]')

    axes[0, 3].set_xlabel('Time [s]')
    axes[0, 3].grid()
    handler1, label1 = axes[0, 3].get_legend_handles_labels()
    axes[0, 3].legend(handler1, label1, loc='upper right')

    axes[1, 3].plot(time, D_roll_target(time), 'gold', label='D_roll_target [deg]')

    axes[1, 3].set_xlabel('Time [s]')
    axes[1, 3].grid()
    handler1, label1 = axes[1, 3].get_legend_handles_labels()
    axes[1, 3].legend(handler1, label1, loc='upper right')


    plt.show()

