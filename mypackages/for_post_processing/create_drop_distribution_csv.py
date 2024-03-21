from mypackages.for_pre_processing import Constants


import pandas as pd
import os

OPEN_PARACHUTE = int(Constants.simulator()['Open_Parachute'])
LOUNCHER_ANGLE = Constants.launcher()['Angle']

class Create_csv:

    def __init__(self, path, wind_ang):
        self.path = path
        self.wind_ang = wind_ang      

    def create_csv(self, wind_ang):
        parachute = ["Never","Apex","Timer"]
        path_results_safetyzone = f"{self.path}/Drop_Distribution_csv/louncher_{LOUNCHER_ANGLE}deg_parachute_{parachute[OPEN_PARACHUTE]}.csv"
        columns = ["風速","風向","経度","緯度"]

        li = []
        for wind, ang in wind_ang:
            path_excel = f"{self.path}/Summaries_History/Summary_wind_{wind}_ang_{ang}.xlsx"
            df_Wind = pd.read_excel(path_excel, sheet_name="Wind", usecols=["wind","angle"])
            df_Landing = pd.read_excel(path_excel, sheet_name="Landing", usecols=["longitude", "latitude"])
            li_Wind = df_Wind.values.tolist()
            li_Landing = df_Landing.values.tolist()
            li_merge = li_Wind + li_Landing
            li_e = sum(li_merge, [])
            li.append(li_e)

        df = pd.DataFrame(li, columns=columns)
        df.to_csv(path_results_safetyzone, mode="w", index=False , encoding="shift-jis")