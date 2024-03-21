import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from itertools import product

import ray

import sys
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == '__main__':
    from for_pre_processing.input_from_Setting import  Constants_Class_from_Copy_Setting

else:
    from mypackages.for_pre_processing.input_from_Setting import Constants
    Constants_default = Constants


class Create_Graphs:

    def __init__(self, path):

        self.path = Path(path)
        self.path_save = self.path / 'Graphs'
        self.path_Copy_Setting = Path(self.path) / 'Copy_Setting'

        if __name__ == '__main__':
            Constants = Constants_Class_from_Copy_Setting(self.path_Copy_Setting)
        else:
            Constants = Constants_default

        self.LAUNCHER_ANGLE = Constants.launcher()['Angle']

        WIND_STD_INIT = Constants.wind()['Power_law']['Wind_STD_Init']
        INTERVAL_WIND_STD = Constants.wind()['Power_law']['Interval_Wind_STD']
        VARIATION_WIND_STD = int(Constants.wind()['Power_law']['Variation_Wind_STD'])
        WIND_AZIMUTH_INIT = Constants.wind()['Power_law']['Wind_Azimuth_Init']
        VARIATION_WIND_AZIMUTH = int(Constants.wind()['Power_law']['Variation_Wind_Azimuth'])
        
        if WIND_STD_INIT == 0.0 and VARIATION_WIND_AZIMUTH != 1:

            self.wind_ang_list = list(product(np.arange(WIND_STD_INIT + INTERVAL_WIND_STD, WIND_STD_INIT + VARIATION_WIND_STD * INTERVAL_WIND_STD, INTERVAL_WIND_STD), np.linspace(WIND_AZIMUTH_INIT, 360, VARIATION_WIND_AZIMUTH + 1)[:-1]))
            self.wind_ang_list.insert(0, (0.0, float(WIND_AZIMUTH_INIT)))

        else:

            self.wind_ang_list = list(product(np.arange(WIND_STD_INIT, WIND_STD_INIT + VARIATION_WIND_STD * INTERVAL_WIND_STD, INTERVAL_WIND_STD), np.linspace(WIND_AZIMUTH_INIT, 360, VARIATION_WIND_AZIMUTH + 1)[:-1]))

    def get_colors(self, number, cmap_name, wind_num):
        Rs = []
        Gs = []
        Bs = []
        cm = plt.get_cmap(cmap_name, wind_num) # 風の数だけ分割
        i = 0
        for i in range(wind_num):
            Rs.append(cm(i)[0])
            Gs.append(cm(i)[1])
            Bs.append(cm(i)[2])
        return (Rs[number], Gs[number], Bs[number])

    def create_graph(self, wind_ang):

        wind, ang = wind_ang

        #Excelファイルを読み込む
        path_excel = self.path / f'Histories/wind_{wind}_ang_{ang}.xlsx'
        df = pd.read_excel(str(path_excel))

        #グラフを描画
        plt.plot(df['time'], df['v_x'], label='v_x')
        plt.plot(df['time'], df['v_y'], label='v_y')
        plt.plot(df['time'], df['v_z'], label='v_z')

        #グラフのラベル
        plt.title(f'wind_{wind}_ang_{ang}')
        plt.xlabel('time')
        plt.ylabel('verocity')

        #グラフの表示
        plt.legend()
        #plt.show()

        """
        4_Results/...から指定した風速、角度のエクセルファイルを読み取り、
        指定したx,y軸のグラフを作成し、表示する。
        次のようなことにも対応させること。
        x=時間でy= V_x, v_y, v_z
        のようにして一つのグラフに3本の曲線が描画できると良い。
        """

    #理想フライト
    def create_graphs_ideal_flight(self):

        path_ideal_flight = self.path_save / 'Ideal_Flight'
        path_ideal_flight.mkdir(parents=True, exist_ok=True)

        path_excel = self.path / 'Histories/wind_0.0_ang_0.0.xlsx'

        df_Scalar = pd.read_excel(str(path_excel), sheet_name='Scalar',usecols=["phase", "time","burning","Fst","dp"])
        phase_1_data = df_Scalar[df_Scalar['phase'] == 1]
        phase_2_data = df_Scalar[df_Scalar['phase'] == 2]
        phase_3_data = df_Scalar[df_Scalar['phase'] == 3]
        phase_4_data = df_Scalar[df_Scalar['phase'] == 4]
        Burning_1_data = df_Scalar[df_Scalar['burning'] == 1]
        Burning_0_data = df_Scalar[df_Scalar['burning'] == 0]
        
        # 各フェーズデータの先頭の数の取得
        p1_start = 0
        p2_start = len(phase_1_data)
        p3_start = p2_start + len(phase_2_data)
        end = len(df_Scalar)-1
        if len(phase_4_data)==0:
            p4_start = p3_start
        else:
            p4_start = p3_start + len(phase_3_data)
        
        #(1)Fst
        plt.plot(Burning_1_data['time'], Burning_1_data['Fst'], label='Burning', color='red')
        plt.plot(Burning_0_data['time'], Burning_0_data['Fst'], label='Not Burning', color='c')
        plt.plot(phase_1_data['time'][p1_start], phase_1_data['Fst'][p1_start], marker="$1$", label='Phase 1 start', color='black')
        plt.plot(phase_2_data['time'][p2_start], phase_2_data['Fst'][p2_start], marker="$2$", label='Phase 2 start', color='black')
        plt.plot(phase_3_data['time'][p3_start], phase_3_data['Fst'][p3_start], marker="$3$", label='Phase 3 start', color='black') 
        if len(phase_4_data)== 0:
            plt.legend()
        else:
            plt.plot(phase_4_data['time'][p4_start], phase_4_data['Fst'][p4_start], marker="$4$", label='Phase 4 start', color='black')  
            plt.legend()
        
        plt.title('Fst')
        plt.xlabel('Time[s]')
        plt.ylabel('StaticMargin[%]')

        #グラフに要素を追加
        plt.grid(True)
        y_min, y_max = plt.ylim()
        yticks = range(round(int(y_min), -1)-6, round(int(y_max)+6, -1), 2)
        yticklabels = [f'{value}%' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_ideal_flight / 'Fst_wind_0.0.png'))
        #plt.show
        plt.close()


        #(2)DynamicPressure
        plt.plot(Burning_1_data['time'], Burning_1_data['dp'], label='Burning', color='red')
        plt.plot(Burning_0_data['time'], Burning_0_data['dp'], label='Not Burning', color='c')
        plt.plot(phase_1_data['time'][p1_start], phase_1_data['dp'][p1_start], marker="$1$", label='Phase 1 start', color='black')
        plt.plot(phase_2_data['time'][p2_start], phase_2_data['dp'][p2_start], marker="$2$", label='Phase 2 start', color='black')
        plt.plot(phase_3_data['time'][p3_start], phase_3_data['dp'][p3_start], marker="$3$", label='Phase 3 start', color='black') 
        if len(phase_4_data)==0:
            plt.legend()
        else:
            plt.plot(phase_4_data['time'][p4_start], phase_4_data['dp'][p4_start], marker="$4$", label='Phase 4 start', color='black') 
            plt.legend()

        #グラフのラベル
        plt.title('DynamicPressure')
        plt.xlabel('Time[s]')
        plt.ylabel('DP[kPa]')

        #グラフに要素を追加
        plt.grid(True)
        y_min, y_max = plt.ylim()
        yticks = range(round(int(y_min), -1), round(int(y_max), -1), 2)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_ideal_flight / 'DP_wind_0.0.png'))
        #plt.show
        plt.close()

        #(3)Body Velocity
        df_Body = pd.read_excel(str(path_excel), sheet_name='Body',usecols=["time","airflow_x","airflow_y","airflow_z"])

        # グラフを描画
        plt.plot(df_Body['time'], df_Body['airflow_x'], label='v_x')
        plt.plot(df_Body['time'], df_Body['airflow_y'], label='v_y')
        plt.plot(df_Body['time'], df_Body['airflow_z'], label='v_z')
        y_min, y_max = plt.ylim()
        plt.vlines(df_Body["time"][p2_start], y_min, y_max, colors="gray", linestyles="dotted")
        plt.vlines(df_Body["time"][p3_start], y_min, y_max, colors="gray", linestyles="dotted")
        plt.text((df_Body["time"][p1_start]+df_Body["time"][p2_start])/2, y_max, "1", ha="center")
        plt.text((df_Body["time"][p2_start]+df_Body["time"][p3_start])/2, y_max, "2", ha="center")
        if len(phase_4_data)==0:
            plt.text((df_Body["time"][p3_start]+df_Body["time"][end])/2, y_max, "3", ha="center")
        else:
            plt.vlines(df_Body["time"][p4_start], y_min, y_max, colors="gray", linestyles="dotted")        
            plt.text((df_Body["time"][p3_start]+df_Body["time"][p4_start])/2, y_max, "3", ha="center")
            plt.text((df_Body["time"][p4_start]+df_Body["time"][end])/2, y_max, "4", ha="center")


        #グラフのラベル
        plt.title('Body Velocity')
        plt.xlabel('Time[s]')
        plt.ylabel('Body Velocity[m/s]')

        #グラフに要素を追加
        plt.grid(True)
        plt.legend()
        yticks = range(round(int(y_min), -1), round(int(y_max)+10, -1), 10)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_ideal_flight / 'Body_Velocity_wind_0.0.png'))
        #plt.show
        plt.close()


        #(4)Trajectory
        df_Launcher = pd.read_excel(str(path_excel), sheet_name='Launcher',usecols=['phase','burning','DownRange','r_x'])
        phase_1_data = df_Launcher[df_Launcher['phase'] == 1]
        phase_2_data = df_Launcher[df_Launcher['phase'] == 2]
        phase_3_data = df_Launcher[df_Launcher['phase'] == 3]
        phase_4_data = df_Launcher[df_Launcher['phase'] == 4]
        Burning_1_data = df_Launcher[df_Launcher['burning'] == 1]
        Burning_0_data = df_Launcher[df_Launcher['burning'] == 0]

        #グラフを描画
        plt.plot(Burning_1_data['DownRange'], Burning_1_data['r_x'], label='Burning', color='red')
        plt.plot(Burning_0_data['DownRange'], Burning_0_data['r_x'], label='Not Burning', color='c')
        plt.plot(phase_1_data['DownRange'][p1_start], phase_1_data['r_x'][p1_start], marker="$1$", label='Phase 1 start', color='black')
        plt.plot(phase_2_data['DownRange'][p2_start], phase_2_data['r_x'][p2_start], marker="$2$", label='Phase 2 start', color='black')
        plt.plot(phase_3_data['DownRange'][p3_start], phase_3_data['r_x'][p3_start], marker="$3$", label='Phase 3 start', color='black') 
        if len(phase_4_data)==0:
            plt.legend()
        else:
            plt.plot(phase_4_data['DownRange'][p4_start], phase_4_data['r_x'][p4_start], marker="$4$", label='Phase 4 start', color='black') 
            plt.legend()
        
        #グラフのラベル
        plt.title('Trajectory')
        plt.xlabel('DownRange[m]')
        plt.ylabel('Altitude[m]')

        #グラフに要素を追加
        plt.grid(True)
        y_min, y_max = plt.ylim()
        yticks = range(round(int(y_min), -1), round(int(y_max), -1), 100)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_ideal_flight / 'Trajectory_wind_0.0.png'))
        #plt.show
        plt.close()


    #ノミナルフライト
    @ray.remote
    def create_graphs_nominal_flight(self, wind_ang):

        wind, ang = wind_ang

        path_nominal_flight = self.path_save / 'Nominal_Flight' / f'wind_{wind}_ang_{ang}'
        path_nominal_flight.mkdir(parents=True, exist_ok=True)

        path_excel = self.path / f'Histories/wind_{wind}_ang_{ang}.xlsx'


        #(1)Wind Speed
        df_GND_Wind = pd.read_excel(str(path_excel), sheet_name='GND_Wind',usecols=['phase','burning','wind_norm','Altitude'])
        Burning_1_data = df_GND_Wind[df_GND_Wind['burning'] == 1]
        Burning_0_data = df_GND_Wind[df_GND_Wind['burning'] == 0]
        phase_1_data = df_GND_Wind[df_GND_Wind['phase'] == 1]
        phase_2_data = df_GND_Wind[df_GND_Wind['phase'] == 2]
        phase_3_data = df_GND_Wind[df_GND_Wind['phase'] == 3]
        phase_4_data = df_GND_Wind[df_GND_Wind['phase'] == 4]

        # 各フェーズデータの先頭の数の取得
        p1_start = 0
        p2_start = len(phase_1_data)
        p3_start = p2_start + len(phase_2_data)
        end = len(df_GND_Wind)-1
        if len(phase_4_data)==0:
            p4_start = p3_start
        else:
            p4_start = p3_start + len(phase_3_data)

        #グラフを描画
        plt.plot(df_GND_Wind['wind_norm'], df_GND_Wind['Altitude'], color='c')
        #グラフのラベル
        plt.title('Wind Speed')
        plt.xlabel('Wind Speed[m/s]')
        plt.ylabel('Altitude[m]')

        #グラフに要素を追加
        plt.grid(True)
        y_min, y_max = plt.ylim()
        yticks = range(0, round(int(y_max), -2), 200)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_nominal_flight / f'Wind_Speed_wind_{wind}_ang_{ang}.png'))
        #plt.show
        plt.close()

        #(2)Acceleration
        df_Align = pd.read_excel(str(path_excel), sheet_name='Align',usecols=['time','a_x_G','a_y_G','a_z_G','phase'])

        #グラフを描画
        plt.plot(df_Align['time'], df_Align['a_x_G'], label='a_x')
        plt.plot(df_Align['time'], df_Align['a_y_G'], label='a_y')
        plt.plot(df_Align['time'], df_Align['a_z_G'], label='a_z')
        y_min, y_max = plt.ylim()
        plt.vlines(df_Align["time"][p2_start], y_min, y_max, colors="gray", linestyles="dotted")
        plt.vlines(df_Align["time"][p3_start], y_min, y_max, colors="gray", linestyles="dotted")
        plt.text((df_Align["time"][p1_start]+df_Align["time"][p2_start])/2, y_max, "1", ha="center")
        plt.text((df_Align["time"][p2_start]+df_Align["time"][p3_start])/2, y_max, "2", ha="center")
        if len(phase_4_data)==0:
            plt.text((df_Align["time"][p3_start]+df_Align["time"][end])/2, y_max, "3", ha="center")
        else:
            plt.vlines(df_Align["time"][p4_start], y_min, y_max, colors="gray", linestyles="dotted")        
            plt.text((df_Align["time"][p3_start]+df_Align["time"][p4_start])/2, y_max, "3", ha="center")
            plt.text((df_Align["time"][p4_start]+df_Align["time"][end])/2, y_max, "4", ha="center")


        #グラフのラベル
        plt.title('Acceleration')
        plt.xlabel('Time[s]')
        plt.ylabel('Body Acceleration[G]')

        #グラフに要素を追加
        plt.grid(True)
        plt.legend()
        yticks = range(round(int(y_min), -1), round(int(y_max), -1), 2)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_nominal_flight / f'Acceleration_wind_{wind}_ang_{ang}.png'))
        #plt.show
        plt.close()

        #(3)DynamicPressure
        #Excelファイルを読み込む
        df_Scalar = pd.read_excel(str(path_excel), sheet_name='Scalar',usecols=['phase','burning','time','dp'])
        phase_1_data = df_Scalar[df_Scalar['phase'] == 1]
        phase_2_data = df_Scalar[df_Scalar['phase'] == 2]
        phase_3_data = df_Scalar[df_Scalar['phase'] == 3]
        phase_4_data = df_Scalar[df_Scalar['phase'] == 4]
        Burning_1_data = df_Scalar[df_Scalar['burning'] == 1]
        Burning_0_data = df_Scalar[df_Scalar['burning'] == 0]
        
        #グラフを描画
        plt.plot(Burning_1_data['time'], Burning_1_data['dp'], label='Burning', color='red')
        plt.plot(Burning_0_data['time'], Burning_0_data['dp'], label='Not Burning', color='c')
        plt.plot(phase_1_data['time'][p1_start], phase_1_data['dp'][p1_start], marker="$1$", label='Phase 1 start', color='black')
        plt.plot(phase_2_data['time'][p2_start], phase_2_data['dp'][p2_start], marker="$2$", label='Phase 2 start', color='black')
        plt.plot(phase_3_data['time'][p3_start], phase_3_data['dp'][p3_start], marker="$3$", label='Phase 3 start', color='black')        
        if len(phase_4_data)==0:
            plt.legend()
        else:   
            plt.plot(phase_4_data['time'][p4_start], phase_4_data['dp'][p4_start], marker="$4$", label='Phase 4 start', color='black')      
            plt.legend()
        
        #グラフのラベル
        plt.title('DynamicPressure')
        plt.xlabel('Time[s]')
        plt.ylabel('DP[kPa]')

        #グラフに要素を追加
        plt.grid(True)
        y_min, y_max = plt.ylim()
        yticks = range(round(int(y_min), -1), round(int(y_max), -1), 2)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)
        
        #グラフをGraphsに保存
        plt.savefig(str(path_nominal_flight / f'DP_wind_{wind}_ang_{ang}.png'))
        #plt.show
        plt.close()

        #(4)Body Velocity
        #Excelファイルを読み込む
        df_Body = pd.read_excel(str(path_excel), sheet_name='Body',usecols=['time','airflow_x','airflow_y','airflow_z'])

        #グラフを描画
        plt.plot(df_Body['time'], df_Body['airflow_x'], label='v_x')
        plt.plot(df_Body['time'], df_Body['airflow_y'], label='v_y')
        plt.plot(df_Body['time'], df_Body['airflow_z'], label='v_z')
        y_min, y_max = plt.ylim()
        plt.vlines(df_Body["time"][p2_start], y_min, y_max, colors="gray", linestyles="dotted")
        plt.vlines(df_Body["time"][p3_start], y_min, y_max, colors="gray", linestyles="dotted")
        plt.text((df_Body["time"][p1_start]+df_Body["time"][p2_start])/2, y_max, "1", ha="center")
        plt.text((df_Body["time"][p2_start]+df_Body["time"][p3_start])/2, y_max, "2", ha="center")
        if len(phase_4_data)==0:
            plt.text((df_Body["time"][p3_start]+df_Body["time"][end])/2, y_max, "3", ha="center")
        else:
            plt.vlines(df_Body["time"][p4_start], y_min, y_max, colors="gray", linestyles="dotted")
            plt.text((df_Body["time"][p3_start]+df_Body["time"][p4_start])/2, y_max, "3", ha="center")
            plt.text((df_Body["time"][p4_start]+df_Body["time"][end])/2, y_max, "4", ha="center")
        
        #グラフのラベル
        plt.title('Body Velocity')
        plt.xlabel('Time[s]')
        plt.ylabel('Body Velocity[m/s]')
        
        #グラフに要素を追加
        plt.grid(True)
        plt.legend()
        yticks = range(round(int(y_min), -1), round(int(y_max)+10, -1), 10)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_nominal_flight / f'Body_Velocity_{wind}_ang_{ang}.png'))
        #plt.show
        plt.close()

        #(5)Trajectory
        df_Launcher = pd.read_excel(str(path_excel), sheet_name='Launcher',usecols=['phase','burning','DownRange','r_x'])
        phase_1_data = df_Launcher[df_Launcher['phase'] == 1]
        phase_2_data = df_Launcher[df_Launcher['phase'] == 2]
        phase_3_data = df_Launcher[df_Launcher['phase'] == 3]
        phase_4_data = df_Launcher[df_Launcher['phase'] == 4]
        Burning_1_data = df_Launcher[df_Launcher['burning'] == 1]
        Burning_0_data = df_Launcher[df_Launcher['burning'] == 0]
                
        #グラフを描画
        plt.plot(Burning_1_data['DownRange'], Burning_1_data['r_x'], label='Burning', color='red')
        plt.plot(Burning_0_data['DownRange'], Burning_0_data['r_x'], label='Not Burning', color='c')
        plt.plot(phase_1_data['DownRange'][p1_start], phase_1_data['r_x'][p1_start], marker="$1$", label='Phase 1 start', color='black')
        plt.plot(phase_2_data['DownRange'][p2_start], phase_2_data['r_x'][p2_start], marker="$2$", label='Phase 2 start', color='black')
        plt.plot(phase_3_data['DownRange'][p3_start], phase_3_data['r_x'][p3_start], marker="$3$", label='Phase 3 start', color='black') 
        if len(phase_4_data)==0:
            plt.legend()
        else:
            plt.plot(phase_4_data['DownRange'][p4_start], phase_4_data['r_x'][p4_start], marker="$4$", label='Phase 4 start', color='black') 
            plt.legend()
        
        #グラフのラベル
        plt.title('Trajectory')
        plt.xlabel('Downrange[m]')
        plt.ylabel('Altitude[m]')

        #グラフに要素を追加
        plt.grid(True)
        y_min, y_max = plt.ylim()
        yticks = range(round(int(y_min), -1), round(int(y_max), -1), 100)
        yticklabels = [f'{value}' for value in yticks]
        plt.yticks(yticks, labels=yticklabels)

        #グラフをGraphsに保存
        plt.savefig(str(path_nominal_flight / f'Trajectory_{wind}_ang_{ang}.png'))
        #plt.show
        plt.close()

        """
        4_Results/...から審査書に求められる風速、角度のエクセルファイルを読み取り、
        指定したx,y軸のグラフを作成し、Graphsに保存する。この際きれいなグラフを心掛けること
        例
        グラフのアスペクト比
        タイトル
        凡例
        線の太さ、色
        """


    def create_3dgraph_wind(self, wind_ang, fig=None, ax=None):
        if any([fig is None, ax is None]):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            is_animation = False

        else:
            fig, ax = fig, ax
            is_animation = True

        wind_list = [wind_ang[0] for wind_ang in wind_ang]

        wind_elements = list(set(wind_list))
        wind_num = len(wind_elements)

        wind_ang_elements = [list(filter(lambda x: x[0] == wind, wind_ang))[0] for wind in wind_elements]

        colors = [self.get_colors(i,"jet", wind_num=wind_num) for i in range(wind_num)]

        for wind_ang in wind_ang:
            wind, ang = wind_ang
            #Excelファイルを読み込む
            path_excel = self.path / f'Histories/wind_{wind}_ang_{ang}.xlsx'
            df_GND_raw = pd.read_excel(str(path_excel), "GND", usecols=["phase", "burning", "r_x", "r_y", "r_z"])

            arr_GND = df_GND_raw.to_numpy()
            arr_r = arr_GND[:, 2:5]
            arr_r = -1*arr_r
            arr_GND[:, 2:5] = arr_r
            df_GND = pd.DataFrame(data=arr_GND, columns=df_GND_raw.columns)

            # 3Dグラフを描画
            if wind_ang in wind_ang_elements:
                ax.plot(df_GND['r_x'], df_GND['r_y'], df_GND['r_z'], color=colors[int(wind)], label=f'{wind} m/s')

            else:
                ax.plot(df_GND['r_x'], df_GND['r_y'], df_GND['r_z'], color=colors[int(wind)])

        # 軸ラベルの設定
        ax.set_xlabel('E-W [m]')
        ax.set_ylabel('N-S [m]')
        ax.set_zlabel('Altitude [m]')

        # グラフにタイトルを追加
        ax.set_title(f'{self.LAUNCHER_ANGLE} deg')
        ax.legend(loc='upper left', bbox_to_anchor=(1., 1.1))

        if not is_animation:
            #グラフをGraphsに保存
            #plt.savefig(str(self.path_save/f'3D_Trajectory_Wind'))

            # グラフを表示
            plt.show()


    def create_rotating_3dgraph_wind(self, wind_ang):
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        graph = self.create_3dgraph_wind(wind_ang, fig, ax)

        def rotate_graph3d(angle):
            ax.view_init(azim=angle*1.5)

        ani = FuncAnimation(fig, func=rotate_graph3d, frames=240, init_func=graph, interval=100)
        ani.save(str(self.path_save/f'3D_Trajectory_Wind.gif'), writer='pillow')


    def create_3dgraph_burning(self, wind_ang):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for wind_ang in wind_ang:
            #Excelファイルを読み込む
            wind, ang = wind_ang
            path_excel = self.path / f'Histories/wind_{wind}_ang_{ang}.xlsx'
            df_GND_raw = pd.read_excel(str(path_excel), "GND", usecols=["phase", "burning", "r_x", "r_y", "r_z"])

            arr_GND = df_GND_raw.to_numpy()
            arr_r = arr_GND[:, 2:5]
            arr_r = -1*arr_r
            arr_GND[:, 2:5] = arr_r
            df_GND = pd.DataFrame(data=arr_GND, columns=df_GND_raw.columns)

            phase_1_data = df_GND[df_GND['phase'] == 1]
            phase_2_data = df_GND[df_GND['phase'] == 2]
            phase_3_data = df_GND[df_GND['phase'] == 3]
            phase_4_data = df_GND[df_GND['phase'] == 4]
            Burning_1_data = df_GND[df_GND['burning'] == 1]
            Burning_0_data = df_GND[df_GND['burning'] == 0]

            p1_start = 0
            p2_start = len(phase_1_data)
            p3_start = p2_start + len(phase_2_data)
            end = len(df_GND)-1
            if len(phase_4_data)==0:
                p4_start = p3_start
            else:
                p4_start = p3_start + len(phase_3_data)

            # 3Dグラフを描画
            ax.plot(Burning_1_data['r_x'], Burning_1_data['r_y'], Burning_1_data['r_z'], label='Burning', color='red')
            ax.plot(Burning_0_data['r_x'], Burning_0_data['r_y'], Burning_0_data['r_z'], label='Not Burning', color='c')
            #ax.plot(phase_1_data['r_x'][p1_start], phase_1_data['r_y'][p1_start], phase_1_data['r_z'][p1_start] , marker="$1$", label='Phase 1 start', color='black')
            #ax.plot(phase_2_data['r_x'][p2_start], phase_2_data['r_y'][p2_start], phase_2_data['r_z'][p2_start] , marker="$2$", label='Phase 2 start', color='black')
            ax.plot(phase_3_data['r_x'][p3_start], phase_3_data['r_y'][p3_start], phase_3_data['r_z'][p3_start] , marker="$3$", label='Phase 3 start', color='black')
            if len(phase_4_data)==0:
                #ax.legend()
                pass
            else:
                #ax.plot(phase_4_data['r_x'][p4_start], phase_4_data['r_y'][p4_start],phase_4_data['r_z'][p4_start] ,marker="$4$", label='Phase 4 start', color='black')
                #ax.legend()
                pass

        # 軸ラベルの設定
        ax.set_xlabel('E-W [m]')
        ax.set_ylabel('N-S [m]')
        ax.set_zlabel('Altitude [m]')

        # グラフにタイトルを追加
        ax.set_title(f'{self.LAUNCHER_ANGLE} deg')

        #グラフをGraphsに保存
        #plt.savefig(str(self.path_save/f'3D_Trajectory_Burning'))


        # グラフを表示
        #plt.show()

        def rotate_graph3d(self, angle):
            ax.view_init(azim=angle*6)

        def create_rotate_graph3d(self):
            ani = FuncAnimation()

if __name__ == '__main__':
    Create_Graphs = Create_Graphs(r'C:\Users\genia\00_Oshima\4_Results\2024-0229-170643_70_np_nc_CFD')
    #Create_Graphs.create_3dgraph_wind(Create_Graphs.wind_ang_list)
    Create_Graphs.create_rotating_3dgraph_wind(Create_Graphs.wind_ang_list)