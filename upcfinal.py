import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


V = 628.3  # Volume of the space station in m^3 
Cd = 0.7  # Discharge coefficient, typical for choked flow
gamma = 1.4  # Adiabatic index for air
R = 287  # Specific gas constant for air in J/(kg*K)
T = 293  # Temperature in Kelvin (20°C)
initial_pressure = 101300  # Initial pressure in Pa (1 atm)
final_pressure = 0.3 * 101300  # Final pressure in Pa (0.3 atm)

hole_diameters = [0.004, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02, 0.05, 0.1] 

def mass_flow_rate(P, A):
    return  ((Cd*A*P)/np.sqrt(T))*np.sqrt((gamma/R))*(2/((gamma+1)))**(((gamma+1)/(2*(gamma-1))))
    
def pressure_change(t, P, A):
    if P <= final_pressure:
        return 0  
    mass_loss_rate = mass_flow_rate(P, A)
    dP_dt = -(mass_loss_rate * R * T) / V 
    return dP_dt

time_span = (0, 500000)
time_points = np.linspace(0, time_span[1], 500)

results = []
plt.figure(figsize=(12, 8))

for hole_diameter in hole_diameters:
    A = np.pi * (hole_diameter / 2) ** 2

    solution = solve_ivp(pressure_change, time_span, [initial_pressure], args=(A,), dense_output=True)

    pressure_values = solution.sol(time_points)[0]
    pressure_values_atm = pressure_values / 101300 
    plt.plot(time_points, pressure_values_atm, label=f"{hole_diameter * 100:.1f} cm")
    if np.any(pressure_values_atm <= 0.3):
        time_to_final_pressure = time_points[np.where(pressure_values_atm <= 0.3)[0][0]]
    else:
        time_to_final_pressure = None  
    results.append((hole_diameter * 100, time_to_final_pressure))  

plt.axhline(y=0.3, color='r', linestyle='--', label="Target pressure (0.3 atm)")
plt.xlabel("Time (seconds)")
plt.ylabel("Pressure (atm)")
plt.title("Pressure Drop in Space Station for Various Hole Sizes")
plt.ylim(0, 1.1)
plt.xlim(0, time_span[1])
plt.legend(title="Hole Diameter", bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.grid()
plt.tight_layout()  
plt.show()

results_df = pd.DataFrame(results, columns=["Hole Diameter (cm)", "Time to Reach 0.3 atm (s)"])
results_df["Time to Reach 0.3 atm (days)"] =results_df["Time to Reach 0.3 atm (s)"]/3600
print(results_df)

