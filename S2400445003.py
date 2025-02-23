import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting (Source: https://matplotlib.org/stable/contents.html)
import argparse  # Argparse for command-line argument parsing (Source: https://docs.python.org/3/library/argparse.html)

# Source: Bakker et al., "Tyre Modelling for Use in Vehicle Dynamics Studies", 1987
def bakker_tyre_model(slip, alpha, weight, mu):
    """
    Compute side and braking forces using the Bakker tyre model.
    :param slip: Longitudinal slip (0-1 scale)
    :param alpha: Slip angle (degrees)
    :param weight: Normal force (total vehicle weight)
    :param Fz: Normal force (vehicle weight on one wheel)
    :param mu: Friction coefficient
    :return: Side force and braking force
    """
    # Convert alpha angle to radians
    alpha_rad = np.radians(alpha)
    
    # Bakker coefficients from Table 2 (Source: Bakker et al., 1987)
    a1_side, a2_side, a3_side, a4_side, a5_side, a6_side, a7_side, a8_side = -22.2, 1011, 1078, 1.82, 0.208, 0, -0.354, 0.707
    a1_brake, a2_brake, a3_brake, a4_brake, a5_brake, a6_brake, a7_brake, a8_brake = -21.3, 1144, 49.6, 226, 0.069, -0.006, 0.056, 0.486

    #kN per wheel
    Fz = (weight*9.8)/(4*1000) #kN per wheel
    
    # Peak factor D (Eq. 6 in Bakker et al.)
    D_side = (a1_side * Fz**2 + a2_side * Fz) * mu # From Eq. 6
    D_brake = (a1_brake * Fz**2 + a2_brake * Fz) * mu # From Eq. 6
    
    # C for side force
    C_side = 1.30 # From Eq. 9
    # C for brake force    
    C_brake = 1.65 # From Eq. 9
    
    # Stiffness factor B side force
    BCD_side = a3_side * np.sin(a4_side * np.arctan(Fz * a5_side)) # From Eq. 7
    B_side = BCD_side / (C_side * D_side)  # From Eq. 10
    
    # Stiffness factor B brake force
    BCD_brake = (a3_brake * Fz**2 + a4_brake * Fz) / np.exp(a5_brake * Fz)  # From Eq. 8
    B_brake = BCD_brake / (C_brake * D_brake)  # From Eq. 10
    
    # Load-dependent curvature factor E
    E_side = a6_side * Fz**2 + a7_side * Fz + a8_side # From Eq. 11
    E_brake = a6_brake * Fz**2 + a7_brake * Fz + a8_brake # From Eq. 11
    
    # Side force calculation (Fixed with combined slip dependency)
    phi_side = (1 - E_side) * (slip*2*np.pi) + (E_side / B_side) * np.arctan(B_side * (slip*2*np.pi))
    side_force = D_side * np.sin(C_side * np.arctan(B_side * phi_side))
    
    # Braking force calculation (Fixed with combined slip dependency)
    phi_brake = (1 - E_brake) * (alpha_rad + slip*2*np.pi) + (E_brake / B_brake) * np.arctan(B_brake * (slip*2*np.pi + alpha_rad))
    brake_force = D_brake * np.sin(C_brake * np.arctan(B_brake * phi_brake))
    
    return side_force, brake_force

# Source: Bakker et al., "Tyre Modelling for Use in Vehicle Dynamics Studies", 1987
def plot_forces(weight, mu, alpha_values):
    """
    Generate and save plots for side and braking forces over longitudinal slip.
    :param weight: Normal force (total vehicle weight)
    :param mu: Friction coefficient
    :param alpha_values: List of slip angles to consider
    """
    slip_values = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 5))
    
    # Plot side force
    plt.subplot(1, 2, 1)
    for alpha in alpha_values:
        side_forces = [bakker_tyre_model(s, alpha, weight, mu)[0] for s in slip_values]
        plt.plot(slip_values * 100, side_forces, label=f"α={alpha}°")
    plt.xlabel("Longitudinal Slip (%)")
    plt.ylabel("Side Force (N)")
    plt.title("Side Force vs. Longitudinal Slip")
    plt.legend()
    
    # Plot braking force
    plt.subplot(1, 2, 2)
    for alpha in alpha_values:
        brake_forces = [bakker_tyre_model(s, alpha, weight, mu)[1] for s in slip_values]
        plt.plot(slip_values * 100, brake_forces, label=f"α={alpha}°")
    plt.xlabel("Longitudinal Slip (%)")
    plt.ylabel("Braking Force (N)")
    plt.title("Braking Force vs. Longitudinal Slip")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("tyre_forces_plot.png")
    plt.show()

def main():
    """ Parse command-line arguments and run the simulation. """
    parser = argparse.ArgumentParser(description="Compute and plot tyre forces.")
    parser.add_argument("--weight", type=float, required=True, help="Weight of vehicle (kg)")
    parser.add_argument("--mu", type=float, required=True, help="Friction coefficient")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0, 5, 10, 15],
                        help="List of slip angles in degrees (default: 0, 5, 10, 15)")
    
    args = parser.parse_args()
    plot_forces(args.weight, args.mu, args.alphas)

if __name__ == "__main__":
    main()
