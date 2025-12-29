import PullData as pld
import VizData as vd
import SDOF_modal_anlysis as sdf
import interpolate_data as idf
import VizMode as vm
import comparaison_method as cm
import polymax as pm
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm 
import argparse
from joblib import Parallel, delayed


number_data     = 1
number_data_set = str(number_data).zfill(5)
name_data       = f"../data/first_lab/DPsv{number_data_set}.mat"
name_set        = f"set_{number_data}"

data      = pld.extract_data(name_data)
H1_2 = data["H1_2"][:, 1]

cmif      = sdf.compute_cmif(data)
freq_cmif = np.real(data["H1_2"][:, 0])

def experimental_modal_analys() :
    print("Experimental modal analysis is running")

    vd.bode_plot(data,name_set)
    vd.coherence_plot(data,name_set)
    vd.plot_exitasion_shock(data,name_set)
    vd.plot_time_shock(data,name_set)
    vd.plot_accelerometer_time(data,name_set)
    vd.cmf_plot(freq_cmif, cmif,name_set)

    freq_first_lab  = np.real(data["H1_2"][:, 0])
    mask            = (freq_first_lab >= 18.4) & (freq_first_lab <= 19.4)
    freq_first_mode = freq_first_lab[mask]
    H1_2            = data["H1_2"][:, 1]
    H1_2_first_mode = H1_2[mask]
    cmif_first_mode = cmif[mask]

    H1_2_first_mode_abs = np.abs(H1_2_first_mode)


    lin_freq, lin_H  = idf.compute_linear_interp(freq_first_mode, H1_2_first_mode_abs, 1000)

    damping_peak_picking_method = sdf.compute_peak_picking_method(lin_H, lin_freq, plot=True, set_name=name_set)
    print(f"Damping factor for pick picking method cubic: {damping_peak_picking_method}")
    damping_circle_fit_method = sdf.compute_circle_fit_method(freq_first_mode, H1_2_first_mode, plot=True, set_name=name_set)
    print(f"Damping factor for circle fit method: {damping_circle_fit_method}")


array1       = np.arange(1, 29) 
array2       = np.arange(31, 59) 
array3       = np.arange(61, 79)  
result_array = np.concatenate((array1, array2, array3))

H, freq = pld.extract_H_general(result_array)
delta_t = 1.9531 * 10**(-3) 
modal   = {}

def stabilisation_diagram() :


    def compute_one_pole(i, H_gen, freq, delta_t):
        w_i, damping_i, eigenval = pm.get_polymax(H_gen, freq, i, delta_t)

        idx = (w_i/2/np.pi >= 0) & (w_i/2/np.pi <= 180)
        _, idx_unique = np.unique(w_i[idx], return_index=True)

        return i, {
            "wn"      : w_i[idx][idx_unique],
            "zeta"    : damping_i[idx][idx_unique],
            "stable"  : ["x" for _ in range(len(idx_unique))],
            "eigenval": eigenval[idx][idx_unique],
        }
    orders = list(range(20, 100))

    results = Parallel(n_jobs=-1)(
        delayed(compute_one_pole)(i, H, freq, delta_t)
        for i in tqdm(orders, desc="Polymax poles", unit="Poles")
    )

    modal = {i: data for i, data in results}
    dic_order = pm.get_stabilisation(modal)
    vd.viz_stabilisation_diagram(dic_order, cmif, freq_cmif)


def modal_analysis_with_comparaison() :
    print("Modal analysis and compaison with Samcef is running")
    mode_samcef = pld.extract_samcef_shock()
    for i in range(mode_samcef.shape[0]):
        vm.representation_mode(mode_samcef[i],nbr = i, amplifactor=50, samcef= True)
    lambda_pole = np.array([-0.41919794 -118.39096976j, -1.00311643 -252.09465353j,
                        -3.0413183  +551.98202178j, -2.06192248 +563.29930716j,
                        -0.65715752 -609.91674992j, -0.31543238 +660.98898914j,
                        -3.14291266 -741.49124913j, -3.11370721 -786.73446431j,
                        -3.29741211 -789.05887636j, -5.49805437 -816.49573024j,
                        -7.47931026 +849.93239504j, -5.91860892 -900.81265828j,
                        -1.41441498+1044.67998079j])

    a = pm.compute_lsfd(lambda_pole, freq, H)

    mode      = pm.extract_eigenmode(a)
    vd.viz_argand(mode)

    abs_mode  = np.abs(mode)
    sign      = np.sign(np.cos(np.angle(mode)))
    real_mode = abs_mode * sign

    for i in range(real_mode.shape[0]):
        real_mode[i]       = real_mode[i] / np.max(np.abs(real_mode[i]))
        real_mode[i,0]    *=  1
        real_mode[i,1:28] *= -1
        real_mode[i,28:56]*= -1
        real_mode[i,56:62]*= -1
        real_mode[i,62:68]*= -1
        real_mode[i,68:74]*=  1
    for i in range(real_mode.shape[0]):
        vm.representation_mode(real_mode[i],nbr = i, amplifactor=50)
    MAC         = cm.get_modal_assurance_criterion(mode_samcef, real_mode)
    auto_MAC    = cm.get_autoMAC(real_mode)
    vd.viz_MAC(MAC)
    vd.viz_MAC_auto(auto_MAC, samcef = False)
    print("All the figures are saved in the folder figures")

def main():
    parser = argparse.ArgumentParser(description="Experimental Modal Analysis Toolkit")
    parser.add_argument(
        '-m', '--modal_analysis', action='store_true', help="Run only the modal analysis (CMIF, peak_picking method & cirecle fit method)."
    )
    parser.add_argument(
        '-s', '--stabilisation', action='store_true', help="Run only the stabilisation diagram."
    )
    parser.add_argument(
        '-d', '--detailed_analysis', action='store_true', help="Run only the detailed analysis. With take into acount the mode representation and comparaison with the Samcef"
    )

    args = parser.parse_args()

    if not (args.modal_analysis or args.stabilisation or args.detailed_analysis):
        print("No arguments provided, running all tasks.")
        experimental_modal_analys()
        stabilisation_diagram()
        modal_analysis_with_comparaison()
    else :

        if args.modal_analysis:
            experimental_modal_analys()
        elif args.stabilisation:
            stabilisation_diagram()
        elif args.detailed_analysis:
            modal_analysis_with_comparaison()

if __name__ == "__main__":
    main()