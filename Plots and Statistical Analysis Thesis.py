

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:22:02 2023

@author: zaxari
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress

# Define a list of EOSs
eos_list = ['MDI1', 'Ska', 'W', 'WFF1', 'APR1', 'MDI4', 'BGP','HHJ1','SKI4', 'l95','MDI2','MDI3','WFF2', 'HHJ2','BL1','BL2','lattimer_stiff', 'NLD','DH','lattimer_intermediate']  # Add your EOS names here
# Define a custom color palette with shades of blue, pink, purple, and red
custom_colors = [
    '#0099cc', '#ff66b2', '#9900cc', '#ff6666', '#3366ff', '#ff99cc', '#6633cc',
    '#ff3333', '#33ccff', '#ff6699', '#cc0066', '#00cc00', 'g', '#9999ccfb',
    '#ffcc00', '#ccff66', '#ff9900', '#ff6600', '#ff3300', '#13cfcf'
]  # Add more colors if needed

# Define a list of linestyles for each EOS
linestyles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--']  # Customize line styles as needed

# Initialize lists to store data for each EOS
r_1_4_msun = []
lambda_1_4_msun = []
k2_1_4_msun = []
masses_list = []
radii_list = []
k2_values_list = []
lambda_values_list = []
yr_values_list = []
beta_values = []  # New list to store β values
# Loop through each EOS
for i, eos in enumerate(eos_list):
    # Load data from the JSON file for the current EOS
    with open(os.getcwd() + f"\\data_{eos}.json", 'r') as json_file:
        minmax_data = json.load(json_file)

    minmax = np.array(minmax_data)

    # Extract data for plotting
    radii = minmax[:, 0]  # Radius in km
    masses = minmax[:, 1]  # Mass in Msun
    k2_values = minmax[:, 4]  # k2 value
    lambda_values = minmax[:, 5]  # Lambda (λ) values
    yr_values = minmax[:, 3]  # YR values (update if necessary)
    

    # Find the radius corresponding to 1.4M☉ for the current EOS
    index_1_4_msun = np.argmin(np.abs(masses - 1.4))
    r_1_4_msun.append(radii[index_1_4_msun])
    lambda_1_4_msun.append(lambda_values[index_1_4_msun])
    k2_1_4_msun.append(k2_values[index_1_4_msun])
    
    
    # Calculate β for the current EOS
    beta = 1.474 * masses / radii  # β = (1.474 * M) / R
    beta_values.append(beta.tolist())  # Store β values

    # Store data for each EOS
    masses_list.append(masses)
    radii_list.append(radii)
    k2_values_list.append(k2_values)
    lambda_values_list.append(lambda_values)
    yr_values_list.append(yr_values)



# Create a dictionary to store β values for each EOS
beta_data = {eos_list[i]: beta_values[i] for i in range(len(eos_list))}

# Save β values to a JSON file
with open("beta_values.json", "w") as beta_file:
    json.dump(beta_data, beta_file)


# Adjust the line thickness for all the plots
line_thickness = 1.0

# Plot k2-β for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(beta_values[i], k2_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i], linewidth=line_thickness)

plt.xlabel('β')
plt.ylabel('k2')
plt.title('k2-β Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/k2-beta_relation.png')  # Save the graph
plt.show()

# Plot λ-β for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(beta_values[i], lambda_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i], linewidth=line_thickness)
plt.ylim([0,8])
plt.xlabel('β')
plt.ylabel('Lambda (λ)')
plt.title('Lambda-β Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/lambda-beta_relation.png')  # Save the graph
plt.show()
    


# Plot M-R for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(radii_list[i], masses_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i],linewidth=line_thickness)

plt.xlim([8,20])
plt.xlabel('Radius (km)')
plt.ylabel('Mass (Msun)')
plt.title('Mass-Radius (M-R) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/M-R_relation.png')  # Save the graph
plt.show()

# Plot M-R for each EOS
plt.figure(figsize=(10, 6))
with open("GW170817_MR1_maxmass_contour_data.pkl","rb") as f:
    ligo1 = pickle.load(f)
with open("GW170817_MR2_maxmass_contour_data.pkl","rb") as f:
    ligo2 = pickle.load(f)
ligo1['z_grid']=np.where(ligo1['z_grid']!=0.,ligo1['z_grid'],np.nan)
ligo2['z_grid']=np.where(ligo2['z_grid']!=0.,ligo2['z_grid'],np.nan)
#trimming invalid values so sorting is quicker
X1 = list(filter(lambda x: (not np.isnan(x)), ligo1['z_grid'].flatten()))
X2 = list(filter(lambda x: (not np.isnan(x)), ligo2['z_grid'].flatten()))
#function for getting the value of probability density where the confidence % cuts-off
def getProbCutoff(lst, confidence):
    lst.sort(reverse=True)
    S = sum(lst)
    csum = 0
    out = 0
    for i in lst:
        csum += i
        #print("sum:",csum," percentage:",csum/S," i:", i)
        if csum>confidence*S:
            out = i
            break
    return out
#plots probability density
plt.pcolormesh(ligo1['xx'], ligo1['yy'], ligo1['z_grid'], shading='auto', cmap=plt.cm.PuBu)
plt.pcolormesh(ligo2['xx'], ligo2['yy'], ligo2['z_grid'], shading='auto', cmap=plt.cm.YlOrBr)
#plots 90% confidence region boundary
plt.contour(ligo1['xx'], ligo1['yy'], ligo1['z_grid'], [getProbCutoff(X1, 0.5)],linestyles = "dashed", linewidths = .5, colors=["w"])
plt.contour(ligo2['xx'], ligo2['yy'], ligo2['z_grid'], [getProbCutoff(X2, 0.5)],linestyles = "dashed", linewidths = .5, colors=["w"])
#plots 50% confidence region boundary
plt.contour(ligo1['xx'], ligo1['yy'], ligo1['z_grid'], [getProbCutoff(X1, 0.9)], linewidths = .5, colors=["gray"])
plt.contour(ligo2['xx'], ligo2['yy'], ligo2['z_grid'], [getProbCutoff(X2, 0.9)], linewidths = .5, colors=["gray"])
for i, eos in enumerate(eos_list):
    plt.plot(radii_list[i], masses_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i],linewidth=line_thickness)

plt.xlim([8,20])
plt.xlabel('Radius (km)')
plt.ylabel('Mass (Msun)')
plt.title('Mass-Radius (M-R) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/M-R_relation.png')  # Save the graph
plt.show()

# Plot λ-M for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(masses_list[i], lambda_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i],linewidth=line_thickness)

plt.ylim([0,8])
plt.xlabel('Mass (Msun)')
plt.ylabel('Lambda (λ)')
plt.title('Lambda-Mass (λ-M) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/lambda-M_relation.png')  # Save the graph
plt.show()

# Plot k2-M for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(masses_list[i], k2_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i],linewidth=line_thickness)

plt.xlabel('Mass (Msun)')
plt.ylabel('k2')
plt.title('k2-Mass (k2-M) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/k2-M_relation.png')  # Save the graph
plt.show()

# Plot λ-R for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(radii_list[i], lambda_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i],linewidth=line_thickness)

plt.xlim([8,20])
plt.ylim([0,8])
plt.xlabel('Radius (km)')
plt.ylabel('Lambda (λ)')
plt.title('Lambda-Radius (λ-R) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/lambda-R_relation.png')  # Save the graph
plt.show()

# Plot k2-R for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(radii_list[i], k2_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i],linewidth=line_thickness)

plt.xlim([8,20])
plt.xlabel('Radius (km)')
plt.ylabel('k2')
plt.title('k2-Radius (k2-R) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/k2-R_relation.png')  # Save the graph
plt.show()


# Plot y_R-M for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(masses_list[i], yr_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i], linewidth=line_thickness)

plt.xlabel('Mass (Msun)')
plt.ylabel('yr')
plt.title('yr-Mass (yr-M) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/yr-M_relation.png')  # Save the graph
plt.show()

# Plot y_R-R for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(radii_list[i], yr_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i], linewidth=line_thickness)
plt.xlim([8,20])
plt.xlabel('Radius (km)')
plt.ylabel('yr')
plt.title('yr-Radius (yr-R) Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/yr-R_relation.png')  # Save the graph
plt.show()

# Plot y_R-β for each EOS
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.plot(beta_values[i], yr_values_list[i], label=eos, color=custom_colors[i], linestyle=linestyles[i], linewidth=line_thickness)

plt.xlabel('β')
plt.ylabel('yr')
plt.title('yr-β Relation for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/yr-beta_relation.png')  # Save the graph
plt.show()


# Plot R(1.4M☉)-λ for each EOS with custom colors and point labels
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.scatter(lambda_1_4_msun[i], r_1_4_msun[i], label=eos, color=custom_colors[i])

    

plt.xlabel('Lambda (λ)')
plt.ylabel('Radius R(1.4M☉) (km)')
plt.title('R(1.4M☉) vs. Lambda (λ) for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/R_1_4_msun_vs_lambda.png')  # Save the graph
plt.show()

# Plot R(1.4M☉)-k2 for each EOS with custom colors and point labels
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.scatter(k2_1_4_msun[i], r_1_4_msun[i], label=eos, color=custom_colors[i])



plt.xlabel('k2')
plt.ylabel('Radius R(1.4M☉) (km)')
plt.title('R(1.4M☉) vs. k2 for Neutron Stars')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/R_1_4_msun_vs_k2.png')  # Save the graph
plt.show()


# Create a text file for R(1.4M☉)-λ coordinates with UTF-8 encoding
with open("r_lambda_coordinates.txt", "w", encoding="utf-8") as r_lambda_file:
    r_lambda_file.write("EOS, Lambda (λ), Radius R(1.4M☉) (km)\n")
    for i, eos in enumerate(eos_list):
        coordinates = f"{eos}, {lambda_1_4_msun[i]:.6f}, {r_1_4_msun[i]:.6f}\n"
        r_lambda_file.write(coordinates)

# Create a text file for R(1.4M☉)-k2 coordinates with UTF-8 encoding
with open("r_k2_coordinates.txt", "w", encoding="utf-8") as r_k2_file:
    r_k2_file.write("EOS, k2, Radius R(1.4M☉) (km)\n")
    for i, eos in enumerate(eos_list):
        coordinates = f"{eos}, {k2_1_4_msun[i]:.6f}, {r_1_4_msun[i]:.6f}\n"
        r_k2_file.write(coordinates)


# Calculate Pearson's correlation coefficient and p-value for R(1.4M☉)-k2
pearson_corr_k2, pearson_p_value_k2 = pearsonr(r_1_4_msun, k2_1_4_msun)

# Calculate Spearman's rank correlation coefficient and p-value for R(1.4M☉)-k2
spearman_corr_k2, spearman_p_value_k2 = spearmanr(r_1_4_msun, k2_1_4_msun)

# Calculate Pearson's correlation coefficient and p-value for R(1.4M☉)-λ
pearson_corr_lambda, pearson_p_value_lambda = pearsonr(r_1_4_msun, lambda_1_4_msun)

# Calculate Spearman's rank correlation coefficient and p-value for R(1.4M☉)-λ
spearman_corr_lambda, spearman_p_value_lambda = spearmanr(r_1_4_msun, lambda_1_4_msun)

# Format the results with higher accuracy
accuracy = 6  # Adjust the number of decimal places as needed

print(f"Pearson's correlation coefficient (R(1.4M☉)-k2): {pearson_corr_k2:.{accuracy}f}")
print(f"Pearson's p-value (R(1.4M☉)-k2): {pearson_p_value_k2:.{accuracy}f}")
print(f"Spearman's rank correlation coefficient (R(1.4M☉)-k2): {spearman_corr_k2:.{accuracy}f}")
print(f"Spearman's p-value (R(1.4M☉)-k2): {spearman_p_value_k2:.{accuracy}f}")

print(f"Pearson's correlation coefficient (R(1.4M☉)-λ): {pearson_corr_lambda:.{accuracy}f}")
print(f"Pearson's p-value (R(1.4M☉)-λ): {pearson_p_value_lambda:.{accuracy}f}")
print(f"Spearman's rank correlation coefficient (R(1.4M☉)-λ): {spearman_corr_lambda:.{accuracy}f}")
print(f"Spearman's p-value (R(1.4M☉)-λ): {spearman_p_value_lambda:.{accuracy}f}")

# Define a significance level (alpha)
alpha = 0.05

# Interpretation for R(1.4M☉)-k2
if abs(pearson_corr_k2) >= 0.7 and pearson_p_value_k2 < alpha:
    interpretation_k2 = "Strong positive linear correlation"
elif abs(pearson_corr_k2) >= 0.7 and pearson_p_value_k2 >= alpha:
    interpretation_k2 = "No statistically significant linear correlation"
elif 0.3 <= abs(pearson_corr_k2) < 0.7 and pearson_p_value_k2 < alpha:
    interpretation_k2 = "Moderate positive linear correlation"
elif 0.3 <= abs(pearson_corr_k2) < 0.7 and pearson_p_value_k2 >= alpha:
    interpretation_k2 = "No statistically significant linear correlation"
elif abs(pearson_corr_k2) < 0.3:
    interpretation_k2 = "Weak or no linear correlation"

# Interpretation for R(1.4M☉)-λ
if abs(pearson_corr_lambda) >= 0.7 and pearson_p_value_lambda < alpha:
    interpretation_lambda = "Strong positive linear correlation"
elif abs(pearson_corr_lambda) >= 0.7 and pearson_p_value_lambda >= alpha:
    interpretation_lambda = "No statistically significant linear correlation"
elif 0.3 <= abs(pearson_corr_lambda) < 0.7 and pearson_p_value_lambda < alpha:
    interpretation_lambda = "Moderate positive linear correlation"
elif 0.3 <= abs(pearson_corr_lambda) < 0.7 and pearson_p_value_lambda >= alpha:
    interpretation_lambda = "No statistically significant linear correlation"
elif abs(pearson_corr_lambda) < 0.3:
    interpretation_lambda = "Weak or no linear correlation"

# Print interpretations
print("Interpretation for R(1.4M☉)-k2:", interpretation_k2)
print("Interpretation for R(1.4M☉)-λ:", interpretation_lambda)

# Fit a linear regression model for R(1.4M☉)-λ
slope_lambda, intercept_lambda, r_value_lambda, p_value_lambda, std_err_lambda = linregress(lambda_1_4_msun, r_1_4_msun)
best_fit_lambda = lambda x: slope_lambda * x + intercept_lambda

# Fit a linear regression model for R(1.4M☉)-k2
slope_k2, intercept_k2, r_value_k2, p_value_k2, std_err_k2 = linregress(k2_1_4_msun, r_1_4_msun)
best_fit_k2 = lambda x: slope_k2 * x + intercept_k2

# Plot R(1.4M☉)-λ with best fit line
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.scatter(lambda_1_4_msun[i], r_1_4_msun[i], label=eos, color=custom_colors[i], linewidth=line_thickness)

# Add the best fit line to the plot
x_lambda = np.linspace(min(lambda_1_4_msun), max(lambda_1_4_msun), 100)
plt.plot(x_lambda, best_fit_lambda(x_lambda), color='black', linestyle='--', label='Best Fit Line')

# Display the equation for the best fit line
equation_lambda = f'y = {slope_lambda:.2f}x + {intercept_lambda:.2f}'
plt.text(0.1, 0.9, equation_lambda, transform=plt.gca().transAxes, fontsize=12, color='black')

plt.xlabel('Lambda (λ)')
plt.ylabel('Radius R(1.4M☉) (km)')
plt.title('R(1.4M☉) vs. Lambda (λ) with Best Fit Line')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/R_1_4_msun_vs_lambda_with_best_fit.png')  # Save the graph
plt.show()

# Plot R(1.4M☉)-k2 with best fit line
plt.figure(figsize=(10, 6))
for i, eos in enumerate(eos_list):
    plt.scatter(k2_1_4_msun[i], r_1_4_msun[i], label=eos, color=custom_colors[i], linewidth=line_thickness)

# Add the best fit line to the plot
x_k2 = np.linspace(min(k2_1_4_msun), max(k2_1_4_msun), 100)
plt.plot(x_k2, best_fit_k2(x_k2), color='black', linestyle='--', label='Best Fit Line')

# Display the equation for the best fit line
equation_k2 = f'y = {slope_k2:.2f}x + {intercept_k2:.2f}'
plt.text(0.1, 0.9, equation_k2, transform=plt.gca().transAxes, fontsize=12, color='black')

plt.xlabel('k2')
plt.ylabel('Radius R(1.4M☉) (km)')
plt.title('R(1.4M☉) vs. k2 with Best Fit Line')
plt.grid()
plt.legend(title='EOS')
plt.savefig('plots/R_1_4_msun_vs_k2_with_best_fit.png')  # Save the graph
plt.show()


