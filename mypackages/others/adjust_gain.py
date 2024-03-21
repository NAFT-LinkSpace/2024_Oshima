import numpy as np
import pandas as pd
from control.matlab import *
from matplotlib import pyplot as plt

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from for_pre_processing.input_from_Setting import Constants



df = pd.read_excel(r'C:\Users\genia\00_Oshima\4_Results\2024-0211-173946\Histories\wind_6.0_ang_270.0.xlsx', sheet_name='Control')

phase = df['phase'].values
t_full = df['time'].values
delta_full = df['delta_roll_control'].values #操作量
roll_full = df['Roll'].values #制御量
roll_target_full = df['Roll_target'].values #制御目標

t = t_full[phase==3]
t = t - t[0]

delta, roll, roll_target = delta_full[phase==3], roll_full[phase==3], roll_target_full[phase==3]


# 以下PID制御器更新
ROLL_TARGET = Constants.control()['Roll_Target']

Td = tf([1], [1, 2, 1])
L = Td #プレフィルタ


I = tf([1], [1,0]) #積分
D = tf([1, 0], [0.01, 1]) #微分

(Le, t_p, x_p ) = lsim(L * (1 - Td) / Td, roll, t)# P制御擬似誤差
(LIe, t_i, x_i ) = lsim(L * I * (1 - Td) / Td, roll, t) # I制御擬似誤差
(LDe, t_d, x_d ) = lsim(L * D *(1 - Td) / Td, roll, t) # D制御擬似誤差
(Ldelta, t_delta, x_delta ) = lsim(L, delta, t)


invE_PD = np.linalg.pinv([Le,LDe]) #擬似逆行列
K_PD = Ldelta @ invE_PD

invE_PID = np.linalg.pinv([Le,LIe,LDe]) #擬似逆行列
K_PID = Ldelta @ invE_PID


print(f'PD : Kp = {K_PD[0]}, Kd = {K_PD[1]}')
print(f'PID : Kp = {K_PID[0]}, Ki = {K_PID[1]}, Kd = {K_PID[2]}')