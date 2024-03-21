import pandas as pd
import numpy as np
from itertools import product

from math import atan, log, tan, pi, cos
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import simplekml



#To read json
from PIL import Image, ImageDraw, ImageFont
import os

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from for_pre_processing.input_from_Setting import Constants


WIND_STD_INIT = Constants.wind()['Power_law']['Wind_STD_Init']
INTERVAL_WIND_STD = Constants.wind()['Power_law']['Interval_Wind_STD']
VARIATION_WIND_STD = int(Constants.wind()['Power_law']['Variation_Wind_STD'])
WIND_AZIMUTH_INIT = Constants.wind()['Power_law']['Wind_Azimuth_Init']
VARIATION_WIND_AZIMUTH = int(Constants.wind()['Power_law']['Variation_Wind_Azimuth'])


if WIND_STD_INIT == 0.0 and VARIATION_WIND_AZIMUTH != 1:

    wind_ang = list(product(np.arange(WIND_STD_INIT + INTERVAL_WIND_STD, WIND_STD_INIT + VARIATION_WIND_STD * INTERVAL_WIND_STD, INTERVAL_WIND_STD), np.linspace(WIND_AZIMUTH_INIT, 360, VARIATION_WIND_AZIMUTH + 1)[:-1]))
    wind_ang.append((0.0, WIND_AZIMUTH_INIT))

else:

    wind_ang = list(product(np.arange(WIND_STD_INIT, WIND_STD_INIT + VARIATION_WIND_STD * INTERVAL_WIND_STD, INTERVAL_WIND_STD), np.linspace(WIND_AZIMUTH_INIT, 360, VARIATION_WIND_AZIMUTH + 1)[:-1]))


path_summaries_parent = Path(r'C:\Users\genia\00_Oshima\4_Results\2024-0211-191342') / 'Summaries_History'
path_summary = [path_summaries_parent / f'Summary_wind_{wind}_ang_{ang}.xlsx' for wind, ang in wind_ang]

