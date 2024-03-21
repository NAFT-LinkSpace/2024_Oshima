# mypy: ignore-errors
import numpy as np
from numpy import  deg2rad, rad2deg, power, cross, sin, cos
from numpy.linalg import norm, inv
from scipy.integrate import solve_ivp
import quaternion as qtn

import time
from numba import jit

if __name__ == '__main__':
    import matplotlib.pyplot as plt

from pathlib import Path

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from for_pre_processing.input_from_Setting import Constants
    from for_main_processing import for_calculate_variables as FCV
    from for_main_processing import trans_coordinates as Trans
    from for_main_processing import calculate_variables as CV
    from for_main_processing import calculate_vectors as Vectors

else:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from mypackages.for_pre_processing.input_from_Setting import Constants
    from . import for_calculate_variables as FCV
    from . import trans_coordinates as Trans
    from . import calculate_variables as CV
    from . import calculate_vectors as Vectors


TIME_STEP = Constants.simulator()['Time_Step']
TIME_MAX = Constants.simulator()['Time_Max']
RUN_TIME = Constants.thrust()['Run_Time']
TIME_START_CONTROL = Constants.control()['Time_Start_Control']

LAUNCHER_LENGTH = Constants.launcher()['Length']
Cg_BEFORE = Constants.rocket()['Cg']['Before']
GAP = Constants.rocket()['Gap_Tail_Launcher']
OPEN_PARACHUTE = Constants.simulator()['Open_Parachute']
LAUNCHER_ANGLE = deg2rad(Constants.launcher()['Angle'])

K_p = Constants.control()['Gain']['K_p']
K_i = Constants.control()['Gain']['K_i']
K_d = Constants.control()['Gain']['K_d']


def solve(ode, wind_std, wind_angle, X0, t_start, events):
    time_step = TIME_STEP
    t_max = TIME_MAX

    sol = solve_ivp(ode, (t_start, t_max), X0, args=(wind_std, wind_angle), dense_output=True, events=events, rtol=1.0e-3, method='RK45')

    t_final = sol.t[-1]
    X_final = sol.y[:, -1]

    t = np.arange(t_start, t_final, time_step)
    Xt = sol.sol(t)

    return {'t_final' : t_final, 'X_final' : X_final, 't' : t, 'Xt' : Xt, 'fun_Xt' : sol.sol}


def solve_t_eval(ode, wind_std, wind_angle, X0, t_start, events):
    time_step = TIME_STEP
    t_max = TIME_MAX
    t_eval = np.arange(t_start, t_max, 0.1)

    sol = solve_ivp(ode, (t_start, t_max), X0, args=(wind_std, wind_angle), t_eval = t_eval, dense_output=True, events=events, rtol=1.0e-3, method='RK45')

    t_final = sol.t[-1]
    X_final = sol.y[:, -1]

    t = np.arange(t_start, t_final, time_step)
    Xt = sol.sol(t)

    return {'t_final' : t_final, 'X_final' : X_final, 't' : t, 'Xt' : Xt, 'fun_Xt' : sol.sol}


def launch_clear(t, X, wind_std, wind_angle):

    q0, q1, q2, q3 = X[0:4]
    q = np.quaternion(q0, q1, q2, q3)

    r_launcher = np.array(X[7:10])
    r_align = Trans.launcher_to_align(q, r_launcher)

    REACH_LINE = LAUNCHER_LENGTH + Cg_BEFORE

    return REACH_LINE - r_align[0]

launch_clear.terminal = True
launch_clear.direction = -1


def start_control(t, X, wind_std, wind_angle):

    return TIME_START_CONTROL - t

start_control.terminal = True
start_control.direction = -1


def apogee(t, X, wind_std, wind_angle):

    vx_launcher = X[4] / CV.mass(t)

    return vx_launcher

apogee.terminal = True
apogee.direction = -1


def landing(t, X, wind_std, wind_angle):
    LANDING_LEVEL = Constants.simulator()['Landing_Level']
    SEA_LEVEL = - Constants.launcher()['Altitude']
    GROUND_LEVEL = 0

    if LANDING_LEVEL == 0:
        return X[7] - SEA_LEVEL

    if LANDING_LEVEL == 1:
        return X[7] - GROUND_LEVEL

landing.terminal = True
landing.direction = -1


X0_phase_1 = np.zeros(18)

X0_phase_1[0:4] = cos((LAUNCHER_ANGLE - np.pi / 2) / 2), 0, sin((LAUNCHER_ANGLE - np.pi / 2) / 2), 0
X0_phase_1[4:7] = 0, 0, 0
X0_phase_1[7:10] = (Cg_BEFORE + GAP) * sin(LAUNCHER_ANGLE), 0, (Cg_BEFORE + GAP) * cos(LAUNCHER_ANGLE)
X0_phase_1[10:13] = 0, 0, 0
X0_phase_1[13] = 0
X0_phase_1[14:17] = 0, -(90 - Constants.launcher()['Angle']), 0
X0_phase_1[17] = 0


def phase_1(t, X, wind_std, wind_angle):

    q0, q1, q2, q3 = X[0:4]
    px_launcher, py_launcher, pz_launcher = X[4:7]
    rx_launcher, ry_launcher, rz_launcher = X[7:10]
    ang_vx_align, ang_vy_align, ang_vz_align = X[10:13]
    roll = X[13]
    phi, theta, psi = X[14:17]
    Ie = X[17]


    diff_X = np.zeros_like(X)


    q = np.quaternion(q0, q1, q2, q3)

    diff_Ie = CV.e(t, roll)

    ang_v_align = np.array([ang_vx_align, ang_vy_align, ang_vz_align])
    ang_v_launcher = Trans.align_to_launcher(q, ang_v_align)

    diff_roll = rad2deg(ang_v_align[0])
    diff_phi, diff_theta, diff_psi = rad2deg(ang_v_launcher)
    diff_q = Trans.diff_quaternion(q, ang_v_align)


    v_launcher = np.array([px_launcher, py_launcher, pz_launcher]) / CV.mass(t)
    diff_r_launcher = v_launcher

    wind_launcher = Vectors.wind_launcher(rx_launcher, wind_std, wind_angle)
    airflow_launcher = wind_launcher - v_launcher
    airflow_align = Trans.launcher_to_align(q, airflow_launcher)

    F_align = Trans.launcher_to_align(q, Vectors.gravity_launcher(t)) + Vectors.thrust_align(t) + Vectors.Ft_align(rx_launcher, airflow_align, roll)

    if F_align[0] < 0:
        F_align = np.array([0, 0, 0])

    F_align[1], F_align[2] = 0, 0

    F_launcher = Trans.align_to_launcher(q, F_align)
    diff_p_launcher = F_launcher

    diff_ang_v_align = np.array([0, 0, 0])

    diff_X[0:4] = diff_q.components
    diff_X[4:7] = diff_p_launcher
    diff_X[7:10] = diff_r_launcher
    diff_X[10:13] = diff_ang_v_align
    diff_X[13] = diff_roll
    diff_X[14:17] = diff_phi, diff_theta, diff_psi
    diff_X[17] = diff_Ie

    return diff_X


def phase_2(t, X, wind_std, wind_angle):

    q0, q1, q2, q3 = X[0:4]
    px_launcher, py_launcher, pz_launcher = X[4:7]
    rx_launcher, ry_launcher, rz_launcher = X[7:10]
    ang_vx_align, ang_vy_align, ang_vz_align = X[10:13]
    roll = X[13]
    phi, theta, psi = X[14:17]
    Ie = X[17]


    #返す用のリスト作成
    diff_X = np.zeros_like(X)

    #クオータニオンを定義
    q = np.quaternion(q0, q1, q2, q3)

    diff_Ie = CV.e(t, roll)

    ang_v_align = np.array([ang_vx_align, ang_vy_align, ang_vz_align])#地面から見た角速度
    #座標変換
    ang_v_launcher = Trans.align_to_launcher(q, ang_v_align)#ランチャーからみた角速度

    diff_roll = rad2deg(ang_v_align[0])
    diff_phi, diff_theta, diff_psi = rad2deg(ang_v_launcher)
    diff_q = Trans.diff_quaternion(q, ang_v_align)


    v_launcher = np.array([px_launcher, py_launcher, pz_launcher]) / CV.mass(t)
    diff_r_launcher = v_launcher

    wind_launcher = Vectors.wind_launcher(rx_launcher, wind_std, wind_angle)
    airflow_launcher = wind_launcher - v_launcher
    airflow_align = Trans.launcher_to_align(q, airflow_launcher)

    #並進の運動方程式
    #F=重力、推力、空気力１、空気力２の４項の和
    F_launcher = Vectors.gravity_launcher(t) + Trans.align_to_launcher(q, Vectors.thrust_align(t) + Vectors.Ft_align(rx_launcher, airflow_align, roll) + Vectors.Fn_align(rx_launcher, airflow_align, roll))
    diff_p_launcher = F_launcher

    I = CV.inertia(t)
    diff_ang_v_align = inv(I) @ (Vectors.Tn_align(t, rx_launcher, airflow_align, roll) - cross(ang_v_align, I @ ang_v_align))

    diff_X[0:4] = diff_q.components
    diff_X[4:7] = diff_p_launcher
    diff_X[7:10] = diff_r_launcher
    diff_X[10:13] = diff_ang_v_align
    diff_X[13] = diff_roll
    diff_X[14:17] = diff_phi, diff_theta, diff_psi
    diff_X[17] = diff_Ie

    return diff_X


def phase_3(t, X, wind_std, wind_angle):

    q0, q1, q2, q3 = X[0:4]
    px_launcher, py_launcher, pz_launcher = X[4:7]
    rx_launcher, ry_launcher, rz_launcher = X[7:10]
    ang_vx_align, ang_vy_align, ang_vz_align = X[10:13]
    roll = X[13]
    phi, theta, psi = X[14:17]
    Ie = X[17]

    #返す用のリスト作成
    diff_X = np.zeros_like(X)

    #クオータニオンを定義
    q = np.quaternion(q0, q1, q2, q3)

    diff_Ie = CV.e(t, roll)

    ang_v_align = np.array([ang_vx_align, ang_vy_align, ang_vz_align])#地面から見た角速度
    #座標変換
    ang_v_launcher = Trans.align_to_launcher(q, ang_v_align)#ランチャーからみた角速度

    diff_roll = rad2deg(ang_v_align[0])
    diff_phi, diff_theta, diff_psi = rad2deg(ang_v_launcher)
    diff_q = Trans.diff_quaternion(q, ang_v_align)


    v_launcher = np.array([px_launcher, py_launcher, pz_launcher]) / CV.mass(t)
    diff_r_launcher = v_launcher

    wind_launcher = Vectors.wind_launcher(rx_launcher, wind_std, wind_angle)
    airflow_launcher = wind_launcher - v_launcher
    airflow_align = Trans.launcher_to_align(q, airflow_launcher)

    #並進の運動方程式
    #F=重力、推力、空気力１、空気力２の４項の和
    F_launcher = Vectors.gravity_launcher(t) + Trans.align_to_launcher(q, Vectors.thrust_align(t) + Vectors.Ft_align(rx_launcher, airflow_align, roll) + Vectors.Fn_align(rx_launcher, airflow_align, roll))
    diff_p_launcher = F_launcher

    I = CV.inertia(t)
    e, De= CV.e(t, roll), CV.De(t, diff_roll)

    Servo = CV.Servo()
    delta = Servo.act(t, Servo.delta_roll_control(Ie, e, De))
    diff_ang_v_align = inv(I) @ (Vectors.Tn_align(t, rx_launcher, airflow_align, roll) + Vectors.Tt_align_roll_control(rx_launcher, airflow_align, delta) - cross(ang_v_align, I @ ang_v_align))

    diff_X[0:4] = diff_q.components
    diff_X[4:7] = diff_p_launcher
    diff_X[7:10] = diff_r_launcher
    diff_X[10:13] = diff_ang_v_align
    diff_X[13] = diff_roll
    diff_X[14:17] = diff_phi, diff_theta, diff_psi
    diff_X[17] = diff_Ie

    return diff_X


def phase_4(t, X, wind_std, wind_angle):

    q0, q1, q2, q3 = X[0:4]
    px_launcher, py_launcher, pz_launcher = X[4:7]
    rx_launcher, ry_launcher, rz_launcher = X[7:10]
    ang_vx_align, ang_vy_align, ang_vz_align = X[10:13]
    roll = X[13]
    phi, theta, psi = X[14:17]
    Ie = X[17]


    diff_X = np.zeros_like(X)


    q = np.quaternion(q0, q1, q2, q3)

    diff_Ie = CV.e(t, roll)

    v_launcher = np.array([px_launcher, py_launcher, pz_launcher]) / CV.mass(t)
    diff_r_launcher = v_launcher

    wind_launcher = Vectors.wind_launcher(rx_launcher, wind_std, wind_angle)

    airflow_launcher = wind_launcher - v_launcher
    airflow_align = Trans.parachute_launcher_to_align(v_launcher, wind_launcher, airflow_launcher)

    ang_v_align = np.array([ang_vx_align, ang_vy_align, ang_vz_align])
    ang_v_launcher = Trans.parachute_align_to_launcher(v_launcher, wind_launcher, ang_v_align)

    diff_roll = rad2deg(ang_v_align[0])
    diff_phi, diff_theta, diff_psi = rad2deg(ang_v_launcher)
    diff_q = Trans.diff_quaternion(q, ang_v_align)


    F_launcher = Vectors.gravity_launcher(t) + Trans.parachute_align_to_launcher(v_launcher, wind_launcher,  Vectors.Fd_parachute(rx_launcher, airflow_align))
    diff_p_launcher = F_launcher

    diff_ang_v_align = np.array([0, 0, 0])

    diff_X[0:4] = diff_q.components
    diff_X[4:7] = diff_p_launcher
    diff_X[7:10] = diff_r_launcher
    diff_X[10:13] = diff_ang_v_align
    diff_X[13] = diff_roll
    diff_X[14:17] = diff_phi, diff_theta, diff_psi
    diff_X[17] = diff_Ie

    return diff_X


def solve_all(wind_std, wind_angle):

    Servo = CV.Servo()

    #パラシュート解傘なし
    if OPEN_PARACHUTE == 0:
        #launch clear計算

        #Phase1 : start -> launch clear
        solve_phase_1 = solve(phase_1, wind_std, wind_angle, X0_phase_1, 0, launch_clear)

        #Phase2の初期条件はPhase1の最終時刻の状態
        t_start_phase_2 = solve_phase_1['t_final']
        X0_phase_2 = solve_phase_1['X_final']


        #Phase2 : launch clear -> control start
        solve_phase_2 = solve(phase_2, wind_std, wind_angle, X0_phase_2, t_start_phase_2, start_control)

        #Phase3の初期条件はPhase2の最終時刻の状態
        t_start_phase_3 = solve_phase_2['t_final']
        X0_phase_3 = solve_phase_2['X_final']


        #Phase3 : control start -> apogee
        solve_phase_3 = solve_t_eval(phase_3, wind_std, wind_angle, X0_phase_3, t_start_phase_3, apogee)

        #Phase4の初期条件はPhase3の最終時刻の状態
        t_start_phase_4 = solve_phase_3['t_final']
        X0_phase_4 = solve_phase_3['X_final']


        #Phase4 : apogee -> landing
        solve_phase_4 = solve(phase_2, wind_std, wind_angle, X0_phase_4, t_start_phase_4, landing)


        #Phaseごとの解を配列に保存
        t1, t2, t3, t4 = solve_phase_1['t'], solve_phase_2['t'][1:], solve_phase_3['t'][1:], solve_phase_4['t'][1:]
        Xt1, Xt2, Xt3, Xt4 = solve_phase_1['Xt'], solve_phase_2['Xt'][:, 1:], solve_phase_3['Xt'][:, 1:], solve_phase_4['Xt'][:, 1:]

        print(t3[0])
        print(t3[-1])
        #全てのPhaseの解を保存する配列
        t2_3_4 = np.concatenate([t2, t3, t4], axis=0)
        t = np.concatenate([t1, t2, t3, t4], axis=0)
        Xt = np.concatenate([Xt1, Xt2, Xt3, Xt4], axis=1)

        #以降ではアウトプットのためにデータを整理している。

        q = qtn.as_quat_array(np.array([Xt[0], Xt[1], Xt[2], Xt[3]]).T)
        q1, q2_3_4 = q[:len(t1)].copy(), q[len(t1):].copy()

        ang_v_align = np.array([Xt[10], Xt[11], Xt[12]]).T
        ang_v_launcher = Trans.align_to_launcher(q, ang_v_align)
        ang_v_gnd = Trans.launcher_to_gnd(ang_v_launcher)

        roll = Xt[13]
        phi, theta, psi = Xt[14], Xt[15], Xt[16]

        Ie = Xt[17]
        e = CV.e(t, roll)
        De = CV.De(t, rad2deg(ang_v_align[:, 0]))


        Ie_modified, e_modified, De_modified = Ie.copy(), e.copy(), De.copy()

        Ie_modified[:len(t1) + len(t2)] = 0
        Ie_modified[len(t1) + len(t2) + len(t3):] = 0

        e_modified[:len(t1) + len(t2)] = 0
        e_modified[len(t1) + len(t2) + len(t3):] = 0

        De_modified[:len(t1) + len(t2)] = 0
        De_modified[len(t1) + len(t2) + len(t3):] = 0


        #delta_roll_control_actual = np.zeros_like(t)
        #delta_roll_control_actual[len(t1) + len(t2):len(t1) + len(t2) + len(t3)] = Servo.act(t3, Servo.delta_roll_control(Ie_modified[len(t1) + len(t2):len(t1) + len(t2) + len(t3)], e_modified[len(t1) + len(t2):len(t1) + len(t2) + len(t3)], De_modified[len(t1) + len(t2):len(t1) + len(t2) + len(t3)]))

        delta_roll_control = CV.delta_roll_control(Ie_modified, e_modified, De_modified)

        delta_roll_control_K_p = K_p * e_modified
        delta_roll_control_K_i = K_i * Ie_modified
        delta_roll_control_K_d = K_d * De_modified


        r_launcher = np.array([Xt[7], Xt[8], Xt[9]]).T
        r_gnd = Trans.launcher_to_gnd(r_launcher)
        down_range = np.array([np.sqrt(r_launcher[i, 1]**2 + r_launcher[i, 2]**2) for i in range(len(r_launcher)) ])
        altitude = r_launcher[:, 0]

        v_launcher = np.array([Xt[4] / CV.mass(t), Xt[5] / CV.mass(t), Xt[6] / CV.mass(t)]).T
        v_align = Trans.launcher_to_align(q, v_launcher)
        v_gnd = Trans.launcher_to_gnd(v_launcher)
        v_norm = norm(v_launcher, axis=1)

        wind_launcher = Vectors.wind_launcher(altitude, wind_std, wind_angle)
        wind_align = Trans.launcher_to_align(q, wind_launcher)
        wind_gnd = Trans.launcher_to_gnd(wind_launcher)
        wind_north, wind_east = wind_gnd[:, 0], wind_gnd[:, 1]
        wind_norm = norm(wind_launcher, axis=1)

        WIND_MODEL = Constants.simulator()['Wind_Model']
        if WIND_MODEL == 0:
            wind_azimuth = FCV.Power.wind_azimuth(wind_angle)

        if WIND_MODEL == 1:
            wind_azimuth = FCV.Statistic.wind_azimuth(altitude)


        airflow_launcher = wind_launcher - v_launcher
        airflow_align = wind_align - v_align
        airflow_norm = norm(airflow_launcher, axis=1)

        Fn_align = Vectors.Fn_align(altitude, airflow_align, roll)
        Fn_align[:len(t1)][True] = np.array([0.0, 0.0, 0.0])
        Fn_align2_3_4 = Fn_align[len(t1):].copy()
        Fn_norm = norm(Fn_align, axis=1)

        Ft_align = Vectors.Ft_align(altitude, airflow_align, roll)
        Ft_align1, Ft_align2_3_4 = Ft_align[:len(t1)].copy(), Ft_align[len(t1):].copy()
        Ft_norm = norm(Ft_align, axis=1)

        Tn_align = Vectors.Tn_align(t, altitude, airflow_align, roll)
        Tn_align[:len(t1)][True] = np.array([0, 0, 0])
        Tn_align_y = Tn_align[:, 1]

        Tt_align_roll_control = Vectors.Tt_align_roll_control(altitude, airflow_align, delta_roll_control)
        Tt_align_x_roll_control = Tt_align_roll_control[:, 0]


        g = float(Constants.physics()['g'])
        a_align1 = (Trans.launcher_to_align(q1, Vectors.gravity_launcher(t1)) + Vectors.thrust_align(t1) + Ft_align1) / np.array([CV.mass(t1), CV.mass(t1), CV.mass(t1)]).T
        a_align1[a_align1[:, 0] < 0] = np.array([0.0, 0.0, 0.0])
        a_align1[:, 1][True] = 0.0
        a_align1[:, 2][True] = 0.0

        a_launcher1 = Trans.align_to_launcher(q1, a_align1)
        a_launcher2_3_4 = (Vectors.gravity_launcher(t2_3_4) + Trans.align_to_launcher(q2_3_4, Vectors.thrust_align(t2_3_4) + Ft_align2_3_4 + Fn_align2_3_4)) / np.array([CV.mass(t2_3_4), CV.mass(t2_3_4), CV.mass(t2_3_4)]).T
        a_launcher = np.concatenate([a_launcher1, a_launcher2_3_4])

        a_align = Trans.launcher_to_align(q, a_launcher)
        a_gnd = Trans.launcher_to_gnd(a_launcher)
        a_norm = norm(a_launcher, axis=1)

        a_launcher_G = a_launcher / g
        a_align_G = a_align / g
        a_gnd_G = a_gnd / g
        a_norm_G = a_norm / g

        phase = np.concatenate([np.full_like(t1, 1), np.full_like(t2, 2),  np.full_like(t3, 3),  np.full_like(t4, 4)], axis=0)

        burning = np.zeros_like(t, dtype=int)
        burning[t <= RUN_TIME] = 1

        mass = CV.mass(t)
        I = CV.inertia(t)
        I_xx, I_yy, I_zz = I[:, 0, 0], I[:, 1, 1], I[:, 2, 2]

        thrust = FCV.Thrust.thrust(t)
        Cg = CV.Cg(t)

        alpha_airflow = CV.alpha_airflow(airflow_align)
        alpha = CV.alpha(airflow_align)
        beta = CV.beta(airflow_align)
        gamma = CV.gamma(v_gnd)
        xi = CV.xi(v_gnd)

        Cp = CV.Cp(alpha_airflow, roll)
        Fst = CV.Fst(t, alpha_airflow, roll)

        Cl = CV.Cl(alpha_airflow, roll)
        Cn = CV.Cn(alpha_airflow, roll)
        Cd = CV.Cd(alpha_airflow, roll)
        Ct = CV.Ct(alpha_airflow, roll)

        Cmt_roll_control = CV.Cmt_roll_control(delta_roll_control)
        #Cmt_roll_control = CV.Cmt_roll_control(delta_roll_control_actual)

        air_density = CV.air_density(altitude)
        dp = CV.dp_as_kPa(altitude, airflow_launcher)

        index_launch_clear = len(t1) - 1
        index_Max_axisAcc = np.argmax(a_align_G[:, 0])
        index_Max_Q = np.argmax(dp)
        index_apogee = np.argmax(altitude)
        index_landing = -1

        t_launch_clear, a_norm_launch_clear, airflow_norm_launch_clear = t[index_launch_clear], a_norm[index_launch_clear], airflow_norm[index_launch_clear]
        alpha_launch_clear, beta_launch_clear = alpha[index_launch_clear], beta[index_launch_clear]

        t_Max_axisAcc, altitude_Max_axisAcc, a_align_x_Max_axisAcc = t[index_Max_axisAcc], altitude[index_Max_axisAcc], a_align_G[index_Max_axisAcc, 0]

        t_Max_Q, altitude_Max_Q, airflow_norm_Max_Q, dp_Max_Q = t[index_Max_Q], altitude[index_Max_Q], airflow_norm[index_Max_Q], dp[index_Max_Q]

        t_apogee, altitude_apogee, down_range_apogee, airflow_norm_apogee = t[index_apogee], altitude[index_apogee], down_range[index_apogee], airflow_norm[index_apogee]
        latitude_apogee = Trans.gnd_to_latlon(np.array([r_gnd[index_apogee, 0], r_gnd[index_apogee, 1]]))[0]
        longitude_apogee = Trans.gnd_to_latlon(np.array([r_gnd[index_apogee, 0], r_gnd[index_apogee, 1]]))[1]

        t_landing, down_range_landing = t[index_landing], down_range[index_landing]
        latitude_landing = Trans.gnd_to_latlon(np.array([r_gnd[index_landing, 0], r_gnd[index_landing, 1]]))[0]
        longitude_landing = Trans.gnd_to_latlon(np.array([r_gnd[index_landing, 0], r_gnd[index_landing, 1]]))[1]


    #パラシュート頂点解散あり
    if OPEN_PARACHUTE == 1:

        #Phase1 : start -> launch clear
        solve_phase_1 = solve(phase_1, wind_std, wind_angle, X0_phase_1, 0, launch_clear)

        #Phase2の初期条件はPhase1の最終時刻の状態
        t_start_phase_2 = solve_phase_1['t_final']
        X0_phase_2 = solve_phase_1['X_final']


        #Phase2 : launch clear -> control start
        solve_phase_2 = solve(phase_2, wind_std, wind_angle, X0_phase_2, t_start_phase_2, start_control)

        #Phase3の初期条件はPhase2の最終時刻の状態
        t_start_phase_3 = solve_phase_2['t_final']
        X0_phase_3 = solve_phase_2['X_final']


        #Phase3 : control start -> apogee
        solve_phase_3 = solve(phase_3, wind_std, wind_angle, X0_phase_3, t_start_phase_3, apogee)

        #Phase4の初期条件はPhase3の最終時刻の状態
        t_start_phase_4 = solve_phase_3['t_final']
        X0_phase_4 = solve_phase_3['X_final']


        #Phase4 : apogee -> landing
        solve_phase_4 = solve(phase_4, wind_std, wind_angle, X0_phase_4, t_start_phase_4, landing)



        #Phaseごとの解を配列に保存
        t1, t2, t3, t4 = solve_phase_1['t'], solve_phase_2['t'][1:], solve_phase_3['t'][1:], solve_phase_4['t'][1:]
        Xt1, Xt2, Xt3, Xt4 = solve_phase_1['Xt'], solve_phase_2['Xt'][:, 1:], solve_phase_3["Xt"][:, 1:], solve_phase_4["Xt"][:, 1:]

        #全てのPhaseの解をまとめる配列
        t2_3 = np.concatenate([t2, t3], axis=0)
        t = np.concatenate([t1, t2, t3, t4], axis=0)
        Xt = np.concatenate([Xt1, Xt2, Xt3, Xt4], axis=1)


        #以降ではアウトプットのために結果を整理している。

        #quaternion配列に変換
        q = qtn.as_quat_array(np.array([Xt[0], Xt[1], Xt[2], Xt[3]]).T)
        q1, q2_3 = q[:len(t1)].copy(), q[len(t1):len(t1) + len(t2_3)].copy()

        ang_v_align = np.array([Xt[10], Xt[11], Xt[12]]).T
        ang_v_launcher = Trans.align_to_launcher(q, ang_v_align)
        ang_v_gnd = Trans.launcher_to_gnd(ang_v_launcher)

        roll = Xt[13]
        phi, theta, psi = Xt[14], Xt[15], Xt[16]

        Ie = Xt[17]
        e = CV.e(t, roll)
        De = CV.De(t, rad2deg(ang_v_align[:, 0]))


        Ie_modified, e_modified, De_modified = Ie.copy(), e.copy(), De.copy()

        Ie_modified[:len(t1) + len(t2)] = 0
        Ie_modified[len(t1) + len(t2) + len(t3):] = 0

        e_modified[:len(t1) + len(t2)] = 0
        e_modified[len(t1) + len(t2) + len(t3):] = 0

        De_modified[:len(t1) + len(t2)] = 0
        De_modified[len(t1) + len(t2) + len(t3):] = 0


        #delta_roll_control_actual = np.zeros_like(t)
        #delta_roll_control_actual[len(t1) + len(t2):len(t1) + len(t2) + len(t3)] = Servo.act(t3, Servo.delta_roll_control(Ie_modified[len(t1) + len(t2):len(t1) + len(t2) + len(t3)], e_modified[len(t1) + len(t2):len(t1) + len(t2) + len(t3)], De_modified[len(t1) + len(t2):len(t1) + len(t2) + len(t3)]))

        delta_roll_control = CV.delta_roll_control(Ie_modified, e_modified, De_modified)

        delta_roll_control_K_p = K_p * e_modified
        delta_roll_control_K_i = K_i * Ie_modified
        delta_roll_control_K_d = K_d * De_modified


        r_launcher = np.array([Xt[7], Xt[8], Xt[9]]).T
        r_gnd = Trans.launcher_to_gnd(r_launcher)
        down_range = np.array([np.sqrt(r_launcher[i, 1]**2 + r_launcher[i, 2]**2) for i in range(len(r_launcher)) ])
        altitude = r_launcher[:, 0]
        altitude4 = altitude[len(t1) + len(t2_3):].copy()

        v_launcher = np.array([Xt[4] / CV.mass(t), Xt[5] / CV.mass(t), Xt[6] / CV.mass(t)]).T
        v_launcher4 = v_launcher[len(t1) + len(t2_3):].copy()

        v_align = Trans.launcher_to_align(q, v_launcher)
        v_gnd = Trans.launcher_to_gnd(v_launcher)
        v_norm = norm(v_launcher, axis=1)

        wind_launcher = Vectors.wind_launcher(altitude, wind_std, wind_angle)
        wind_launcher4 = wind_launcher[len(t1) + len(t2_3):].copy()

        wind_align = Trans.launcher_to_align(q, wind_launcher)
        wind_gnd = Trans.launcher_to_gnd(wind_launcher)
        wind_north, wind_east = wind_gnd[:, 0], wind_gnd[:, 1]
        wind_norm = norm(wind_launcher, axis=1)


        WIND_MODEL = Constants.simulator()['Wind_Model']
        if WIND_MODEL == 0:
            wind_azimuth = FCV.Power.wind_azimuth(wind_angle)

        if WIND_MODEL == 1:
            wind_azimuth = FCV.Statistic.wind_azimuth(altitude)


        airflow_launcher = wind_launcher - v_launcher
        airflow_align = wind_align - v_align
        airflow_align4 = airflow_align[len(t1) + len(t2_3):].copy()
        airflow_norm = norm(airflow_launcher, axis=1)

        Fn_align = Vectors.Fn_align(altitude, airflow_align, roll)
        Fn_align[:len(t1)][True] = np.array([0.0, 0.0, 0.0])
        Fn_align2_3 = Fn_align[len(t1):len(t1) + len(t2_3)].copy()
        Fn_norm = norm(Fn_align, axis=1)

        Ft_align = Vectors.Ft_align(altitude, airflow_align, roll)
        Ft_align1, Ft_align2_3 = Ft_align[:len(t1)].copy(), Ft_align[len(t1):len(t1) + len(t2_3)].copy()
        Ft_norm = norm(Ft_align, axis=1)

        Tn_align = Vectors.Tn_align(t, altitude, airflow_align, roll)
        Tn_align[:len(t1)][True] = np.array([0, 0, 0])
        Tn_align_y = Tn_align[:, 1]

        Tt_align_roll_control = Vectors.Tt_align_roll_control(altitude, airflow_align, delta_roll_control)
        Tt_align_x_roll_control = Tt_align_roll_control[:, 0]


        g = float(Constants.physics()['g'])
        a_align1 = (Trans.launcher_to_align(q1, Vectors.gravity_launcher(t1)) + Vectors.thrust_align(t1) + Ft_align1) / np.array([CV.mass(t1), CV.mass(t1), CV.mass(t1)]).T
        a_align1[a_align1[:, 0] < 0] = np.array([0.0, 0.0, 0.0])
        a_align1[:, 1][True] = 0.0
        a_align1[:, 2][True] = 0.0

        a_launcher1 = Trans.align_to_launcher(q1, a_align1)
        a_launcher2_3 = (Vectors.gravity_launcher(t2_3) + Trans.align_to_launcher(q2_3, Vectors.thrust_align(t2_3) + Ft_align2_3 + Fn_align2_3)) / np.array([CV.mass(t2_3), CV.mass(t2_3), CV.mass(t2_3)]).T
        a_launcher4 = (Vectors.gravity_launcher(t4) + Trans.parachute_align_to_launcher(v_launcher4, wind_launcher4, Vectors.Fd_parachute(altitude4, airflow_align4))) / np.array([CV.mass(t4), CV.mass(t4), CV.mass(t4)]).T
        a_launcher = np.concatenate([a_launcher1, a_launcher2_3, a_launcher4])

        a_align = Trans.launcher_to_align(q, a_launcher)
        a_gnd = Trans.launcher_to_gnd(a_launcher)
        a_norm = norm(a_launcher, axis=1)

        a_launcher_G = a_launcher / g
        a_align_G = a_align / g
        a_gnd_G = a_gnd / g
        a_norm_G = a_norm / g

        phase = np.concatenate([np.full_like(t1, 1), np.full_like(t2, 2), np.full_like(t3, 3), np.full_like(t4, 4)], axis=0)

        burning = np.zeros_like(t, dtype=int)
        burning[t <= RUN_TIME] = 1

        mass = CV.mass(t)
        I = CV.inertia(t)
        I_xx, I_yy, I_zz = I[:, 0, 0], I[:, 1, 1], I[:, 2, 2]

        thrust = FCV.Thrust.thrust(t)
        Cg = CV.Cg(t)

        alpha_airflow = CV.alpha_airflow(airflow_align)
        alpha = CV.alpha(airflow_align)
        beta = CV.beta(airflow_align)
        gamma = CV.gamma(v_gnd)
        xi = CV.xi(v_gnd)

        Cp = CV.Cp(alpha_airflow, roll)
        Fst = CV.Fst(t, alpha_airflow, roll)

        Cl = CV.Cl(alpha_airflow, roll)
        Cn = CV.Cn(alpha_airflow, roll)
        Cd = CV.Cd(alpha_airflow, roll)
        Ct = CV.Ct(alpha_airflow, roll)

        Cmt_roll_control = CV.Cmt_roll_control(delta_roll_control)
        #Cmt_roll_control = CV.Cmt_roll_control(delta_roll_control_actual)

        air_density = CV.air_density(altitude)
        dp = CV.dp_as_kPa(altitude, airflow_launcher)

        index_launch_clear = len(t1) - 1
        index_Max_axisAcc = np.argmax(a_align_G[:, 0])
        index_Max_Q = np.argmax(dp)
        index_apogee = np.argmax(altitude)
        index_landing = -1

        t_launch_clear, a_norm_launch_clear, airflow_norm_launch_clear = t[index_launch_clear], a_norm[index_launch_clear], airflow_norm[index_launch_clear]
        alpha_launch_clear, beta_launch_clear = alpha[index_launch_clear], beta[index_launch_clear]

        t_Max_axisAcc, altitude_Max_axisAcc, a_align_x_Max_axisAcc = t[index_Max_axisAcc], altitude[index_Max_axisAcc], a_align_G[index_Max_axisAcc, 0]

        t_Max_Q, altitude_Max_Q, airflow_norm_Max_Q, dp_Max_Q = t[index_Max_Q], altitude[index_Max_Q], airflow_norm[index_Max_Q], dp[index_Max_Q]

        t_apogee, altitude_apogee, down_range_apogee, airflow_norm_apogee = t[index_apogee], altitude[index_apogee], down_range[index_apogee], airflow_norm[index_apogee]
        latitude_apogee = Trans.gnd_to_latlon(np.array([r_gnd[index_apogee, 0], r_gnd[index_apogee, 1]]))[0]
        longitude_apogee = Trans.gnd_to_latlon(np.array([r_gnd[index_apogee, 0], r_gnd[index_apogee, 1]]))[1]

        t_landing, down_range_landing = t[index_landing], down_range[index_landing]
        latitude_landing = Trans.gnd_to_latlon(np.array([r_gnd[index_landing, 0], r_gnd[index_landing, 1]]))[0]
        longitude_landing = Trans.gnd_to_latlon(np.array([r_gnd[index_landing, 0], r_gnd[index_landing, 1]]))[1]


    if OPEN_PARACHUTE == 2:
        raise NotImplementedError

    return {'Launcher' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'DownRange' : down_range, 'r_x' : r_launcher[:, 0], 'r_y' : r_launcher[:, 1], 'r_z' : r_launcher[:, 2],
                'v_norm' : v_norm, 'v_x' : v_launcher[:, 0], 'v_y' : v_launcher[:, 1], 'v_z' : v_launcher[:, 2],
                'a_norm' : a_norm, 'a_x' : a_launcher[:, 0], 'a_y' : a_launcher[:, 1], 'a_z' : a_launcher[:, 2],
                'a_norm_G' : a_norm_G, 'a_x_G' : a_launcher_G[:, 0], 'a_y_G' : a_launcher_G[:, 1], 'a_z_G' : a_launcher_G[:, 2],
                'phi' : phi, 'theta' : theta, 'psi' : psi,
                'omega_x' : ang_v_launcher[:, 0], 'omega_y' : ang_v_launcher[:, 1], 'omega_z' : ang_v_launcher[:, 2],
                'wind_norm' : wind_norm, 'wind_x' : wind_launcher[:, 0], 'wind_y' : wind_launcher[:, 1], 'wind_z' : wind_launcher[:, 2]},

            'Align' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'v_norm' : v_norm, 'v_x' : v_align[:, 0], 'v_y' : v_align[:, 1], 'v_z' : v_align[:, 2],
                'a_norm' : a_norm, 'a_x' : a_align[:, 0], 'a_y' : a_align[:, 1], 'a_z' : a_align[:, 2],
                'a_norm_G' : a_norm_G, 'a_x_G' : a_align_G[:, 0], 'a_y_G' : a_align_G[:, 1], 'a_z_G' : a_align_G[:, 2],
                'omega_x' : ang_v_align[:, 0], 'omega_y' : ang_v_align[:, 1], 'omega_z' : ang_v_align[:, 2],
                'wind_norm' : wind_norm, 'wind_x' : wind_align[:, 0], 'wind_y' : wind_align[:, 1], 'wind_z' : wind_align[:, 2]},

            'GND' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'DownRange' : down_range, 'r_x' : r_gnd[:, 0], 'r_y' : r_gnd[:, 1], 'r_z' : r_gnd[:, 2],
                'v_norm' : v_norm, 'v_x' : v_gnd[:, 0], 'v_y' : v_gnd[:, 1], 'v_z' : v_gnd[:, 2],
                'a_norm' : a_norm, 'a_x' : a_gnd[:, 0], 'a_y' : a_gnd[:, 1], 'a_z' : a_gnd[:, 2],
                'a_norm_G' : a_norm_G, 'a_x_G' : a_gnd_G[:, 0], 'a_y_G' : a_gnd_G[:, 1], 'a_z_G' : a_gnd_G[:, 2],
                'gamma' : gamma, 'xi' : xi},

            'GND_Wind' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'Altitude' : r_launcher[:, 0], 'wind_norm' : wind_norm, 'wind_north' : wind_north, 'wind_east' : wind_east, 'wind_azimuth' : wind_azimuth},

            'Body' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'omega_x' : ang_v_align[:, 0], 'omega_y' : ang_v_align[:, 1], 'omega_z' : ang_v_align[:, 2],
                'Roll' : roll,
                'airflow_norm' : airflow_norm, 'airflow_x' : airflow_align[:, 0], 'airflow_y' : airflow_align[:, 1], 'airflow_z' : airflow_align[:, 2],
                'alpha_airflow' : alpha_airflow, 'alpha' : alpha, 'beta' : beta,
                'I_xx' : I_xx, 'I_yy' : I_yy, 'I_zz' : I_zz},

            'Body_Force' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'thrust' : thrust, 'Fn' : Fn_norm, 'Ft' : Ft_norm},

            'Body_Torque' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'Tn' : Tn_align_y, 'Tt' : Tt_align_x_roll_control},

            'Scalar' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'DownRange' : down_range, 'mass' : mass,
                'Cg' : Cg, 'Cp' : Cp, 'Fst' : Fst,
                'air_density' : air_density, 'dp' : dp},

            'Scalar_Aero' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'Cl' : Cl, 'Cd' : Cd, 'Cn' : Cn, 'Ct' : Ct, 'Cmt_roll_control' : Cmt_roll_control,
                'alpha_airflow' : alpha_airflow, 'alpha' : alpha, 'beta' : beta, 'gamma' : gamma, 'xi' : xi,
                'delta_roll_control' : delta_roll_control},

            'Control' :
                {'time' : t, 'phase' : phase, 'burning' : burning,
                'Roll' : roll, 'Roll_target' : CV.roll_target(t),
                'Ie' : Ie, 'e' : e, 'De' : De,
                'delta_roll_control' : delta_roll_control, 'delta_roll_control_K_p' : delta_roll_control_K_p, 'delta_roll_control_K_i' : delta_roll_control_K_i, 'delta_roll_control_K_d' : delta_roll_control_K_d,
                #'delta_roll_control_actual' : delta_roll_control_actual,
                'Cmt_roll_control' : Cmt_roll_control, 'Tt' : Tt_align_x_roll_control,
                },


            'Summary' :
                {'Wind' :
                    {'wind' : wind_std, 'angle' : wind_angle},

                'Launch_Clear' :
                    {'time' : t_launch_clear, 'a_norm' : a_norm_launch_clear, 'airflow_norm' : airflow_norm_launch_clear,
                    'alpha' : alpha_launch_clear, 'beta' : beta_launch_clear},

                'Max_axisAcc' :
                    {'time' : t_Max_axisAcc, 'altitude' : altitude_Max_axisAcc, 'axisAcc' : a_align_x_Max_axisAcc},

                'Max_Q' :
                    {'time' : t_Max_Q, 'altitude' : altitude_Max_Q, 'airflow_norm' : airflow_norm_Max_Q, 'dp' : dp_Max_Q},

                'Apogee' :
                    {'time' : t_apogee, 'altitude' : altitude_apogee, 'DownRange' : down_range_apogee, 'airflow_norm' : airflow_norm_apogee,
                    'longitude' : longitude_apogee, 'latitude' : latitude_apogee},

                'Landing' :
                    {'time' : t_landing, 'DownRange' : down_range_landing, 'longitude' : longitude_landing, 'latitude' : latitude_landing}}}
"""
for wind_std in range (8):
    for wind_angle in np.linspace(0, 360, 9)[:-2]:
        print(solve_all(wind_std, wind_angle)['Summary']['Landing']['longitude'])
        print(solve_all(wind_std, wind_angle)['Summary']['Landing']['latitude'])
"""


if __name__ == '__main__':
    import pprint
    # Call the solve_all function
    output = solve_all(1.0, 0.0)
    print(output)
"""
launch clear calculation start
launch clear calculation end
time:  0.10852969999999962 s
landing calculation start
landing calculation end
time:  6.7969633 s
output data processing start
output data processing end
time:  6.4271697 s
"""
