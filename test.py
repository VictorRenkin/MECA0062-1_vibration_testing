import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='serif') 
plt.rc('text', usetex=True)  
plt.rcParams.update({
    'font.size': 15,       # Taille de police générale
    'legend.fontsize': 19, # Taille de police pour les légendes
    'axes.labelsize': 24,  # Taille de police pour les étiquettes des axes
})
# Provided data
data = {
    "Mode": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "Numerical (Hz)": [18.83, 40.23, 87.97, 89.69, 97.23, 105.21, 118.00, 124.34, 125.61, 129.88, 134.93, 143.08, 166.37],
    "Experimental (Hz)": [17.84, 39.31, 87.25, 89.69, 96.61, 102.33, 128.01, 137.59, 145.40, 154.49, 172.06, 176.14, 176.79],
    "Relative Error (%)": [5.55, 2.34, 0.83, 0.00, 0.64, 2.81, 7.82, 9.59, 13.62, 15.94, 21.59, 18.76, 5.89]
}

# Create a DataFrame
df = pd.DataFrame(data)
primary_color = (20/255, 45/255, 105/255)  # Convert RGB to matplotlib color format
# Plotting the relative errors
plt.figure(figsize=(12, 6))
plt.bar(df["Mode"], df["Relative Error (%)"], color=primary_color, edgecolor='black', alpha=0.7)

# Adding labels and title
plt.xlabel(r'Mode Number [-]')
plt.ylabel(r'Relative Error [\%]')
# plt.title('Relative Error Between Numerical and Experimental Eigenfrequencies', fontsize=16)
plt.xticks(df["Mode"])
# plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.hlines(5, 0, 14, linestyles='-.', color='darkred', label=r'Acceptable range')
# Display the plot
plt.legend()
plt.xlim(0,14)
plt.tight_layout()
# plt.show()
plt.savefig("relative_error.pdf", format="pdf", bbox_inches='tight')