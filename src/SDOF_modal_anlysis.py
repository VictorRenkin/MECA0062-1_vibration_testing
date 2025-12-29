import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import skg.nsphere as nsphere
import pandas as pd
plt.rc('font', family='serif') 
plt.rc('text', usetex=True)  
plt.rcParams.update({
    'font.size': 17,       # Taille de police générale
    'legend.fontsize': 19, # Taille de police pour les légendes
    'axes.labelsize': 20,  # Taille de police pour les étiquettes des axes
})

def compute_cmif(data) : 
    freq = data["H1_2"][:, 0]
    cmif = np.zeros((len(freq))) # like juste one exitasion
    for i in range(len(freq)) : 
        matrix_H_freq = np.array([
            [data["H1_2"][i, 1]],
            [data["H1_3"][i, 1]],
            [data["H1_4"][i, 1]]
        ])
        _, s, _ = np.linalg.svd(matrix_H_freq)

        cmif[i] = s.T @ s
    return cmif


def compute_peak_picking_method(H, freq, plot=True, set_name="") :
    # Calculer l'amplitude du signal
    amplitude = np.abs(H)    
    # Identifier l'indice du pic principal
    main_peak_index = np.argmax(amplitude)
    main_peak_amplitude = amplitude[main_peak_index]
    main_peak_freq = freq[main_peak_index]
    
    # Calculer le point de demi-puissance
    half_power_point = main_peak_amplitude / np.sqrt(2)
    
    # Trouver les fréquences aux points de demi-puissance autour du pic
    idx_Walpha = np.where(amplitude[:main_peak_index] <= half_power_point)[0][-1] if np.any(amplitude[:main_peak_index] < half_power_point) else main_peak_index
    idx_Wbeta  = np.where(amplitude[main_peak_index:] <= half_power_point)[0][0] + main_peak_index if np.any(amplitude[main_peak_index:] < half_power_point) else main_peak_index
    f_Walpha   = freq[idx_Walpha]
    f_Wbeta    = freq[idx_Wbeta]
    damping = (f_Wbeta - f_Walpha) / (2 * main_peak_freq)
    if plot :
        save_dir = f"../figures/first_lab/{set_name}"
        save_path = f"{save_dir}/peak_method.pdf"
        plt.figure(figsize=(8, 5))
        plt.plot(freq, amplitude, color="darkblue", linewidth=1.5, label="Amplitude")
        plt.vlines([f_Walpha, f_Wbeta], ymin=0, ymax=amplitude[idx_Walpha], color="gray", linestyle="--", linewidth=1)
        
        # Annoter les fréquences omega_a, omega_b et la distance delta omega
        plt.text(f_Walpha - 0.03, half_power_point -1, r'$\omega_a$', ha='center', va='top', fontsize=21)
        plt.text(f_Wbeta + 0.03, half_power_point - 1, r'$\omega_b$', ha='center', va='top', fontsize=21)
        plt.annotate(
        '', 
        xy=(f_Wbeta, half_power_point-1.08), 
        xytext=(f_Walpha, half_power_point-1.08),
        arrowprops=dict(arrowstyle='<->', color='black', lw=1.2)
        )
        plt.text((f_Walpha + f_Wbeta) / 2, half_power_point -1, r'$\Delta \omega$', 
            ha='center', va='bottom', fontsize=14)
        
        # Annoter le pic principal et le point de demi-puissance
        plt.scatter(main_peak_freq, main_peak_amplitude, color="crimson", label="Pic Principal", zorder=5)
        plt.scatter(f_Wbeta, half_power_point, color="#556B2F", label="Demi-puissance", zorder=5)
        plt.scatter(f_Walpha, half_power_point, color="#556B2F", zorder=5)
        plt.hlines(main_peak_amplitude, freq[0], freq[-1], color="black", linestyle=":", linewidth=1.5)
        plt.hlines(half_power_point, freq[0], freq[-1], color="black", linestyle=":", linewidth=1.5)
        
        plt.text(main_peak_freq+ 0.05, main_peak_amplitude + 0.15, r'$H^{max}_{rs(k)}$', va='center', fontsize=18)
        plt.text(f_Wbeta + 0.05, half_power_point + 0.25, r'$\frac{H^{max}_{rs(k)}}{\sqrt{2}}$', va='center', fontsize=21)
        
        plt.ylim(0, main_peak_amplitude + 1)
        plt.xlim(freq[0], freq[-1])

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [g/N]")

        # plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
    return damping


def compute_circle_fit_method(freq, H, plot=True, set_name="") :
    omega = 2 * np.pi * freq
    H     = H/(1j*omega)
    data = pd.DataFrame({
    'np.real(cub_freq)': np.real(H),
    'np.imag(cub_freq)': np.imag(H),
    })
    points = data[['np.real(cub_freq)', 'np.imag(cub_freq)']].to_numpy()
    r, c   = nsphere.nsphere_fit(points)
    t      = np.linspace(0, 2 * np.pi, 1000, endpoint=True)
    x, y   = r * np.cos(t) + c[0], r * np.sin(t) + c[1]
    arg_r_interpolate  = np.argmax(np.abs(x))
    arg_r              = np.argmax(np.abs(H))
    w_r    = 2 * np.pi * freq[arg_r]
    damping_matrix = []
    a_ligne = (c[1] - np.imag(H)[arg_r]) / (c[0] - np.real(H)[arg_r])
    b       = np.imag(H)[arg_r] - a_ligne * np.real(H)[arg_r]
    rc = (x[arg_r_interpolate] -c[0], y[arg_r_interpolate] -c[1])

    for i in range(1, 2) :
        arg_wb = arg_r - i
        arg_wa = arg_r + i
        w_a = 2 * np.pi * freq[arg_wa]
        w_b = 2 * np.pi * freq[arg_wb]    
        ca = (np.real(H)[arg_wa] -c[0], np.imag(H)[arg_wa] - c[1])
        cb = (np.real(H)[arg_wb] -c[0], np.imag(H)[arg_wb] -c[1])
        norme_ra = np.linalg.norm(ca)
        norme_rb = np.linalg.norm(cb)
        norme_rc = np.linalg.norm(rc)
        angle_wb = np.arccos(np.dot(ca, rc) / (norme_ra * norme_rc))
        angle_wa = np.arccos(np.dot(cb, rc) / (norme_rb * norme_rc))
        damping = (w_a**2 - w_b**2)/(2 * w_r * (w_a * np.tan(angle_wa/2) + w_b * np.tan(angle_wb/2)))
        if damping < 0 :
            break # This mean that the arg_wb is not good like the frequency is not perfectly divided by 2 
        y_line_wa = a_ligne * np.real(H)[arg_wa] + b
        y_line_wb = a_ligne * np.real(H)[arg_wb] + b
        if y_line_wa > np.imag(H)[arg_wa] :
            break
        if y_line_wb < np.imag(H)[arg_wb] :
            break
        damping_matrix.append(damping)
        if plot :
            plt.plot([c[0], np.real(H)[arg_wa]], [c[1], np.imag(H)[arg_wa]], color='#4169E1')
            plt.plot([c[0], np.real(H)[arg_wb]], [c[1], np.imag(H)[arg_wb]], color='#4169E1')
            plt.plot([c[0], x[arg_r_interpolate]], [c[1], y[arg_r_interpolate]], color='#4169E1')
            # plt.plot([c[0], op_r[0]], [c[1], op_r[1]], color='#4169E1')
            plt.scatter(np.real(H)[arg_wa], np.imag(H)[arg_wa], color='#4169E1', label=f'w_r= {w_r}')
            plt.scatter(np.real(H)[arg_wb], np.imag(H)[arg_wb], color='#4169E1', label=f'w_r= {w_r}')
            # plt.plot([c[0], np.real(H)[arg_wa]], [c[1], np.imag(H)[arg_wa]], color='black')
            # plt.plot([c[0], np.real(H)[arg_wb]], [c[1], np.imag(H)[arg_wb]], color='black')
    damping_matrix = np.array(damping_matrix)

    if plot :
        t = np.linspace(0, 2 * np.pi, 1000, endpoint=True)
        plt.scatter(data['np.real(cub_freq)'].to_numpy(), data['np.imag(cub_freq)'].to_numpy(), color='#800020')
        plt.scatter(np.real(H)[arg_r], np.imag(H)[arg_r], color='#4169E1', label=f'\omega_r= {w_r}')
        plt.scatter(x[arg_r_interpolate], y[arg_r_interpolate], color='#4169E1', label=f'\omega_r= {w_r}')
        plt.text(x[arg_r_interpolate] -0.004, y[arg_r_interpolate], r"$\omega_r^*$", fontsize=17, color='black')
        plt.text(np.real(H)[arg_r]-0.004,  np.imag(H)[arg_r] - 0.003, r"$\omega_r$", fontsize=17, color='black')


        plt.scatter(c[0], c[1], color='#4169E1', label="Center")
        plt.scatter(np.real(H)[arg_wa], np.imag(H)[arg_wa], color='#4169E1', label=f'w_r= {w_r}')
        plt.text(np.real(H)[arg_wa] + 0.001, np.imag(H)[arg_wa], r"$\omega_a$", fontsize=17, color='black')

        plt.scatter(np.real(H)[arg_wb], np.imag(H)[arg_wb], color='#4169E1', label=f'w_r= {w_r}')
        plt.text(np.real(H)[arg_wb]+0.001, np.imag(H)[arg_wb] - 0.002, r"$\omega_b$", fontsize=17, color='black')
        plt.plot(r * np.cos(t) + c[0], r * np.sin(t) + c[1],  color='#4169E1')
        plt.text(c[0]-0.003,c[1] + 0.001, r"$\theta_a$", fontsize=17, color='black')
        plt.text(c[0]-0.002,c[1] - 0.003, r"$\theta_b$", fontsize=17, color='black')

        # plt.plot([np.real(H)[arg_r], op_r[0]], [np.imag(H)[arg_r], op_r[1]], color='black')
        plt.axis('equal')
        plt.xlabel(r"$\mathrm{Re}(H_{wing})$")
        plt.ylabel(r"$\mathrm{Im}(H_{wing})$")
        # plt.show()
        plt.savefig(f"../figures/first_lab/{set_name}/circle_fit.pdf", format="pdf", dpi=300, bbox_inches='tight')
        plt.close()
    return np.mean(damping_matrix)