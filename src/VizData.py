import matplotlib.pyplot as plt
import numpy as np
import SDOF_modal_anlysis as sdf
import os
from scipy.stats import linregress


plt.rc('font', family='serif') 
plt.rc('text', usetex=True)  
plt.rcParams.update({
    'font.size': 20,        # Taille de police générale
    'legend.fontsize': 15,  # Taille de police pour les légendes
    'axes.labelsize': 24,   # Taille de police pour les étiquettes des axes
    'xtick.labelsize': 25,  # Taille de police pour les ticks sur l'axe x
    'ytick.labelsize': 25,  # Taille de police pour les ticks sur l'axe y
})


def cmf_plot(freq, cmf, set_name):
    save_dir = f"../figures/first_lab/{set_name}"
    save_path = f"{save_dir}/CMIF.pdf"
    
    plt.figure(figsize=(10, 4))
    plt.semilogy(freq, cmf) 
    
    points = [
        (18.83, 18.17),
        (40.23, 29.56),
        (87.97, 4.24),
        (89.69, 28.98),
        (97.23, 13.89),
        (105.21, 12.77),
        (118, 0.32),
        (124.34, 0.24),
        (125.61, 0.92),
        (129.88, 2.06),
        (134.93, 0.2),
        (143.08, 0.084),
        (166.37, 0.0067)
    ]

    bordeaux_color = "#800020"  
    point_size = 10  

    for x, y in points:
        plt.scatter(x, y, color=bordeaux_color, s=point_size, zorder=3)  

    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"CMIF [-]")    
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def bode_plot(data, set_name):
    save_dir = f"../figures/first_lab/{set_name}"
    save_path = f"{save_dir}/Bode_plot.pdf"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    freq = data["G1_1"][:, 0]
    H1_2 = data["H1_2"][:, 1]
    H1_3 = data["H1_3"][:, 1]
    H1_4 = data["H1_4"][:, 1]

    plt.figure(figsize=(10, 4))
    plt.semilogy(freq, np.abs(H1_2), label=r"Wing ")
    plt.semilogy(freq, np.abs(H1_3), label=r"Horizontal Tail")
    plt.semilogy(freq, np.abs(H1_4), label=r"Vertical Tail")
    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"Amplitude [g/N] (dB)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout()
    
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def coherence_plot(data, set_name):
    save_dir = f"../figures/first_lab/{set_name}"
    save_path = f"{save_dir}/Coherence_plot.pdf"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    freq = data["G1_1"][:, 0]
    C1_2 = data["C1_2"][:, 1]
    C1_3 = data["C1_3"][:, 1]
    C1_4 = data["C1_4"][:, 1]

    plt.figure(figsize=(10, 4))
    plt.plot(freq, C1_2, label=r"Wing")
    plt.plot(freq, C1_3, label=r"Horizontal Tail ")
    plt.plot(freq, C1_4, label=r"Vertical Tail")
    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"Magnitude [-]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_exitasion_shock(data, set_name) :
    freq = data["G1_1"][:, 0]
    amplitude = data["G1_1"][:, 1]
    plt.figure(figsize=(10,6))
    plt.plot(freq, amplitude)
    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"Amplitude [N]")
    plt.savefig(f"../figures/first_lab/{set_name}/exitasion_shock.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_shock(data, set_name):
    time = data["X1"][:, 0]
    arg = (time <= 0.08)
    amplitude = data["X1"][:, 1]
    amplitude = amplitude[arg]
    time = time[arg] * 100
    plt.figure(figsize=(10, 6)) 
    plt.plot(time, amplitude)
    plt.xlabel(r"Time [ms]") 
    plt.ylabel(r"Amplitude [N]")
    plt.savefig(f"../figures/first_lab/{set_name}/time_shock.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_accelerometer_time(data, set_name):
    time = data["X2"][:, 0]
    amplitude = data["X2"][:, 1]
    plt.figure(figsize=(10, 9))  
    plt.plot(time, amplitude)
    plt.xlabel(r"Time [s]") 
    plt.ylabel(r"Amplitude [g]")

    plt.savefig(f"../figures/first_lab/{set_name}/time_accelerometer.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def viz_stabilisation_diagram(dic_order, cmif, freq, plot_stabilisation_poles = True):
    fig, ax1 = plt.subplots(figsize=(14, 10))
    ax1.set_xlabel(r"Frequency [Hz]")
    ax1.set_ylabel(r"CMIF [-]", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.semilogy(freq, cmif, color='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Poles [-]", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    for key in dic_order.keys():
        w_i = dic_order[key]["wn"]
        stable = dic_order[key]["stable"]
        for i, w in enumerate(w_i) :
            point_color ='red' if stable[i] == 'd' else 'black'
            if stable[i] == 'x':
                ax2.scatter(w/2/np.pi, key, marker = 'o', s=5, color=point_color)
            elif stable[i] == 'v':
                ax2.scatter(w/2/np.pi, key, marker = stable[i], s=10, facecolors='none',color=point_color)
            else:
                ax2.scatter(w/2/np.pi, key, marker = stable[i], s=20, facecolors='none', color=point_color)
    if plot_stabilisation_poles:
        selected_pole = [30,30,34,37,38,32,70,81,30,70,82,30,36]
        freq_pole     = np.array([18.8371850177749, 40.132836645734095, 87.73661054907369,
        89.67312969606662, 97.53280910557633, 105.18499940664444, 117.87999526512655,
        125.18566897797119,125.65863056930331,130.0322772046283,
        135.10023859504435,143.17255674594142,166.22251319140528])
        if max(dic_order.keys()) <  np.max(np.array(selected_pole)) :
            print("The model cannot caputre all the stabilization pole used")
        else:
            lambda_pole = np.zeros(len(selected_pole), dtype=complex)
            omega       = np.zeros(len(selected_pole))
            damp        = np.zeros(len(selected_pole))
            idx_freq    = np.zeros(len(selected_pole), dtype=int)

            for i in range(len(selected_pole)):
                stable_values = dic_order[selected_pole[i]]["stable"]

                arg_stab = np.where(np.array(stable_values) == 'd')[0]

                if len(arg_stab) == 0:
                    print(f"No stable poles found for pole {i}. Skipping...")
                    continue

                wn_stable       = np.array(dic_order[selected_pole[i]]["wn"])[arg_stab]
                eigenval_stable = np.array(dic_order[selected_pole[i]]["eigenval"])[arg_stab]
                damp_stable     = np.array(dic_order[selected_pole[i]]["zeta"])[arg_stab]

                freq_diff = np.abs(wn_stable / (2 * np.pi) - freq_pole[i])
                idx       = np.argmin(freq_diff)

                lambda_pole[i] = eigenval_stable[idx]
                omega[i]       = wn_stable[idx]
                idx_freq[i]    = arg_stab[idx]
                damp[i]        = damp_stable[idx]
            for i in range(len(selected_pole)) :
                ax2.scatter(dic_order[selected_pole[i]]["wn"][idx_freq[i]]/2/np.pi , selected_pole[i] ,color='green', facecolors='none', s=20,marker='d')
            ax2.scatter(200,20, color='green', label=r'Chossen poles', facecolors='none', s=20,marker='d')
            print("Final omega (Hz):", omega / (2 * np.pi))
            print("lambda pole",lambda_pole)
            print("damp",damp)

    ax2.scatter(200,20, color='red',   label=r'Stabilized', facecolors='none', s=20,marker='d')
    ax2.scatter(200,20, color='black', label=r'Unstabilized', s=20,marker='o',)
    ax2.scatter(200,20, color='black', label=r'Stabilized in frequency (1 \%)', s=20, marker='v', facecolors='none')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    plt.xlim(13, 180)
    plt.savefig("../figures/sec_lab/stabilisation_diagram.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_structure(data_samcef):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data_samcef['X_Coord'], data_samcef['Y_Coord'], data_samcef['Z_Coord'], c='blue', marker='o', s=10)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    plt.show()
    plt.close()


def viz_MAC(MAC_matrix):
    plt.figure(figsize=(10, 8), dpi=300)
    
    cax = plt.imshow(MAC_matrix, cmap='Greys', interpolation='nearest', origin='lower')
    plt.colorbar(cax)

    plt.xticks(range(MAC_matrix.shape[1]), [str(i+1) for i in range(MAC_matrix.shape[1])])
    plt.yticks(range(MAC_matrix.shape[0]), [str(i+1) for i in range(MAC_matrix.shape[0])])
    plt.xlabel(r"Modes Testing [-]")
    plt.ylabel(r"Modes Samcef [-]")

    points_white = [(1, 1), (2, 2), (3, 3), (4,4) , (5, 5), (6,6), (11, 11), (12, 12), (7, 8), (13,11),(11, 11),  (7, 9),(8,10)]
    points_black = [(13, 13), (10,10),(9,9)]

    for i, j in points_white:
        plt.text(j-1, i-1, f"{MAC_matrix[i-1, j-1]:.2f}", ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    for i, j in points_black:
        plt.text(j-1, i-1, f"{MAC_matrix[i-1, j-1]:.2f}", ha='center', va='center', color='black', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("../figures/sec_lab/MAC.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()



def viz_MAC_auto(autoMAC_matrice, samcef=False):
    plt.figure(figsize=(10, 8), dpi=300)  
    cax = plt.imshow(autoMAC_matrice, cmap='Greys', interpolation='nearest', origin='lower')
    plt.colorbar(cax)

    plt.xticks(range(autoMAC_matrice.shape[1]), [str(i + 1) for i in range(autoMAC_matrice.shape[1])])
    plt.yticks(range(autoMAC_matrice.shape[0]), [str(i + 1) for i in range(autoMAC_matrice.shape[0])])

    if samcef:
        plt.xlabel(r"Modes Samcef [-]")
        plt.ylabel(r"Modes Samcef [-]")
    else:
        plt.xlabel(r"Modes Testing [-]")
        plt.ylabel(r"Modes Testing [-]")

    plt.tight_layout()
    points_white = [(1, 1), (2, 2), (3, 3), (4,4) , (5, 5), (6,6), (7, 7), (8, 8), (9, 9), (10,10),(11, 11),  (12,12),(13,13)]
    for i, j in points_white:
        plt.text(j-1, i-1, f"{autoMAC_matrice[i-1, j-1]:.2f}", ha='center', va='center', color='white', fontsize=14, fontweight='bold')

    save_path = "../figures/sec_lab/autoMAC_samcef.pdf" if samcef else "../figures/sec_lab/autoMAC_testing.pdf"
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()


def viz_argand(vectors):
    for i, vec in enumerate(vectors):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')

        modules = [np.abs(v) for v in vec]
        arguments = [np.angle(v) for v in vec]

        ax.scatter(arguments, modules, color="red", label="Points")


        x_cartesian = [m * np.cos(a) for m, a in zip(modules, arguments)]
        y_cartesian = [m * np.sin(a) for m, a in zip(modules, arguments)]
        slope, intercept, _, _, _ = linregress(x_cartesian, y_cartesian)

        max_module = max(modules)
        min_module = min(modules)

        x_line = np.linspace(-max_module, max_module, 500) 
        y_line = slope * x_line + intercept 

        angles_line = np.arctan2(y_line, x_line)
        modules_line = np.sqrt(x_line**2 + y_line**2)

        valid_indices = (modules_line >= min_module) & (modules_line <= max_module)
        angles_line = angles_line[valid_indices]
        modules_line = modules_line[valid_indices]

        ax.plot(angles_line, modules_line, linestyle='-.', color='blue', label="Régression")

        ax.grid(True)  
        ax.set_yticks([])  
        ax.yaxis.grid(False) 
        ax.xaxis.grid(True)  

        path = f"../figures/sec_lab/agran_diagram/argand_{i}.pdf"
        plt.savefig(path, format="pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)  
