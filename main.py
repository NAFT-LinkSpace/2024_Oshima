from mypackages.for_main_processing import solve_ode_full_output
from mypackages import for_pre_processing
from mypackages.for_post_processing import output_to_excel
from mypackages.for_post_processing.plot_drop_distribution import ResultViewer
from mypackages.for_post_processing.plot_graphs import Create_Graphs
from mypackages.for_post_processing.create_drop_distribution_csv import Create_csv
from mypackages.for_pre_processing.input_from_Setting import Constants

import os
from itertools import product
from pprint import pprint
import numpy as np

import ray
ray.init()


WIND_STD_INIT = Constants.wind()['Power_law']['Wind_STD_Init']
INTERVAL_WIND_STD = Constants.wind()['Power_law']['Interval_Wind_STD']
VARIATION_WIND_STD = int(Constants.wind()['Power_law']['Variation_Wind_STD'])
WIND_AZIMUTH_INIT = Constants.wind()['Power_law']['Wind_Azimuth_Init']
VARIATION_WIND_AZIMUTH = int(Constants.wind()['Power_law']['Variation_Wind_Azimuth'])


class Simulator:
    def __init__(self):

        self.path = for_pre_processing.Setting.path()['Results']

        if WIND_STD_INIT == 0.0 and VARIATION_WIND_AZIMUTH != 1:

            self.wind_ang = list(product(np.arange(WIND_STD_INIT + INTERVAL_WIND_STD, WIND_STD_INIT + VARIATION_WIND_STD * INTERVAL_WIND_STD, INTERVAL_WIND_STD), np.linspace(WIND_AZIMUTH_INIT, 360, VARIATION_WIND_AZIMUTH + 1)[:-1]))
            self.wind_ang.append((0.0, float(WIND_AZIMUTH_INIT)))

        else:

            self.wind_ang = list(product(np.arange(WIND_STD_INIT, WIND_STD_INIT + VARIATION_WIND_STD * INTERVAL_WIND_STD, INTERVAL_WIND_STD), np.linspace(WIND_AZIMUTH_INIT, 360, VARIATION_WIND_AZIMUTH + 1)[:-1]))


    def pre(self):
        for_pre_processing.Results.make_dirs()
        for_pre_processing.Setting.save()


    def post(self, wind_ang):

        image = ResultViewer(self.path, wind_ang)
        image.draw_safety_zone()
        image.draw_grid()
        image.draw_landing()
        image.draw_info()
        image.draw_legend()
        image.draw_colorbar()
        image.save("Nominal")

        D_csv = Create_csv(self.path, self.wind_ang)
        D_csv.create_csv(wind_ang)

        graphs = Create_Graphs(self.path)

        if WIND_STD_INIT == 0.0:

            graphs.create_graphs_ideal_flight()

        #for wind_ang in self.wind_ang:
        #    graphs.create_graphs_nominal_flight(wind_ang)

        graphs_tasks = [graphs.create_graphs_nominal_flight.remote(graphs, i) for i in self.wind_ang]
        ray.get(graphs_tasks)


    @ray.remote
    def run_single_simulation(self, wind_ang):
        wind, ang = wind_ang
        out = solve_ode_full_output.solve_all(wind, ang)

        path_results_history = os.path.join(self.path, f"Histories/wind_{wind}_ang_{ang}.xlsx")
        path_results_summary = os.path.join(self.path, f"Summaries_History/Summary_wind_{wind}_ang_{ang}.xlsx")

        output_to_excel.create_history_excel(out, path_results_history)
        output_to_excel.create_summary_excel(out, path_results_summary)

    def run_simulation(self):

        self.pre()

        if Constants.simulator()['Wind_Model'] == 0:

            simulator_tasks = [self.run_single_simulation.remote(self, i) for i in self.wind_ang]
            ray.get(simulator_tasks)

        if Constants.simulator()['Wind_Model'] == 1:
            self.run_single_simulation((0.0, 0.0))

        self.post(self.wind_ang)


if __name__ == "__main__":
    sim = Simulator()
    sim.run_simulation()