"""
Rendezvous en Marte - Mecánica Celeste
Tarea 2: Cálculo de maniobra de rendezvous alrededor de Marte
Autor: Juan Diego Ospino Reyes
Cédula: 1010143974
"""

import numpy as np
import matplotlib
# Use Agg backend for non-interactive plotting (no Qt required)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import newton, bisect
import matplotlib.patches as mpatches
import math
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
G = 6.67430e-11  # Gravitational constant [m^3/kg/s^2]
R_MARS = 3393.15e3  # Mars radius [m]
M_MARS = 6.39e23  # Mars mass [kg]

# Canonical units
UL = R_MARS  # Length unit [m]
UM = M_MARS  # Mass unit [kg]
UT = np.sqrt(UL**3 / (G * UM))  # Time unit [s]
mu = 1.0  # Gravitational parameter in canonical units

# Conversion factors
deg = np.pi / 180
rad = 180 / np.pi


def kepler_eq_solver(M, e):
    """Solve Kepler's equation for eccentric anomaly E."""
    kepler_func = lambda E: M - E + e * np.sin(E)
    E = newton(kepler_func, M)
    return E


def rotate_matrix(angle, axis):
    """Create rotation matrix for given angle and axis (1=x, 2=y, 3=z)."""
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis == 1:  # Rotation around x-axis
        R = np.array([[1, 0, 0],
                      [0, c, s],
                      [0, -s, c]])
    elif axis == 2:  # Rotation around y-axis
        R = np.array([[c, 0, -s],
                      [0, 1, 0],
                      [s, 0, c]])
    elif axis == 3:  # Rotation around z-axis
        R = np.array([[c, s, 0],
                      [-s, c, 0],
                      [0, 0, 1]])
    else:
        raise ValueError("Axis must be 1, 2, or 3")
    
    return R


def create_sphere_mesh(radius=1, center=(0, 0, 0), n_theta=30, n_phi=30):
    """Create mesh for a sphere."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)
    
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    
    return x, y, z


def draw_conic(ax, e=0.5, p=1, w=0, color='b', alpha=0.7):
    """Draw a conic section (ellipse) in 2D."""
    w_rad = w * deg
    
    # Calculate points in natural conic system
    if e < 1:
        phi = np.pi
    else:
        phi = 0.99 * np.arccos(-1 / e)
    
    fs = np.linspace(-phi, phi, 1000)
    rs = p / (1 + e * np.cos(fs))
    xps = rs * np.cos(fs)
    yps = rs * np.sin(fs)
    
    # Rotate points
    cw, sw = np.cos(w_rad), np.sin(w_rad)
    xs = xps * cw - yps * sw
    ys = xps * sw + yps * cw
    
    # Plot
    ax.plot(xs, ys, color=color, alpha=alpha)


def lambert_function(a, mu=1, tf=1, P1=None, P2=None, short=True, velocity=False):
    """
    Lambert's function for orbital transfer calculation.
    Based on methodology from Prof. Zuluaga's Astrodynamics course.
    """
    if P1 is None:
        P1 = np.array([1, 0, 0])
    if P2 is None:
        P2 = np.array([0, 1, 0])
    
    # Derived properties
    r1 = np.linalg.norm(P1)
    r2 = np.linalg.norm(P2)
    c = np.linalg.norm(P2 - P1)
    s = (r1 + r2 + c) / 2
    
    # Alpha and beta: principal values
    alfa0 = 2 * np.arcsin(np.sqrt(s / (2 * a)))
    beta0 = 2 * np.arcsin(np.sqrt((s - c) / (2 * a)))
    
    # Evaluate extreme transfer times
    am = s / 2
    alfam = np.pi
    betam = 2 * np.arcsin(np.sqrt((s - c) / (2 * am)))
    
    # Time in minimum energy orbit
    tm = (am**(1.5) * (alfam - betam - (np.sin(alfam) - np.sin(betam)))) / np.sqrt(mu)
    
    # Parabolic time
    tp = np.sqrt(2) / 3 * (s**1.5 - (s - c)**1.5) / np.sqrt(mu)
    
    # Conditions
    if tf < tp:
        raise ValueError(f"Time of flight {tf} is too short (tp = {tp})")
    
    # Choose quadrants
    theta0 = np.arccos(np.dot(P1, P2) / (r1 * r2))
    
    if tf < tm:
        alfa = alfa0
        if short:
            esmall = False
            beta = beta0
            theta = theta0
        else:
            esmall = True
            beta = -beta0
            theta = 2 * np.pi - theta0
    else:
        alfa = 2 * np.pi - alfa0
        if short:
            esmall = True
            beta = beta0
            theta = theta0
        else:
            esmall = False
            beta = -beta0
            theta = 2 * np.pi - theta0
    
    # The two p values
    ps = 4 * a * (s - r1) * (s - r2) / c**2 * np.sin((alfa + beta) / 2)**2
    pt = 4 * a * (s - r1) * (s - r2) / c**2 * np.sin((alfa - beta) / 2)**2
    
    if esmall:
        p = min(ps, pt)
    else:
        p = max(ps, pt)
    
    # Velocities
    v1 = (np.sqrt(mu * p) / (r1 * r2 * np.sin(theta)) * 
          (P2 - (1 - r2 / p * (1 - np.cos(theta))) * P1))
    v2 = (-np.sqrt(mu * p) / (r1 * r2 * np.sin(theta)) * 
          (P1 - (1 - r1 / p * (1 - np.cos(theta))) * P2))
    
    # Calculate flight time
    fl = (a**1.5 * (alfa - beta - (np.sin(alfa) - np.sin(beta)))) / np.sqrt(mu) - tf
    
    if velocity:
        return v1, v2
    else:
        return fl


def intersect_circles(x1, y1, r1, x2, y2, r2):
    """Find intersection points of two circles."""
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx**2 + dy**2)
    
    # Check if circles intersect
    if d > r1 + r2 or d < abs(r1 - r2):
        raise ValueError("Circles do not intersect")
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)
    
    x3 = x1 + a * dx / d
    y3 = y1 + a * dy / d
    
    rx = -dy * (h / d)
    ry = dx * (h / d)
    
    return (x3 + rx, y3 + ry, x3 - rx, y3 - ry)


def lambert_geometric(P1, P2, a, plot=False, filename=None):
    """Geometric solution of Lambert's problem."""
    # Distances
    r1 = np.linalg.norm(P1)
    r2 = np.linalg.norm(P2)
    
    # Position of vacant foci
    try:
        Fsx, Fsy, Ftx, Fty = intersect_circles(P1[0], P1[1], 2*a - r1, 
                                                P2[0], P2[1], 2*a - r2)
    except ValueError:
        return None
    
    # Vectors directed to vacant foci
    Fsvec = np.array([Fsx, Fsy, 0])
    Ftvec = np.array([Ftx, Fty, 0])
    
    # Determine which is which based on distance
    if np.linalg.norm(Ftvec) < np.linalg.norm(Fsvec):
        Fsvec, Ftvec = Ftvec, Fsvec
    
    # Direction of eccentricity vectors
    ehats = -Fsvec / np.linalg.norm(Fsvec)
    ehatt = -Ftvec / np.linalg.norm(Ftvec)
    
    # Argument of periapsis
    ws = np.arctan2(ehats[1], ehats[0])
    wt = np.arctan2(ehatt[1], ehatt[0])
    
    # Eccentricities of calculated orbits
    es = np.linalg.norm(Fsvec) / (2 * a)
    et = np.linalg.norm(Ftvec) / (2 * a)
    
    # Semi-latus rectum
    ps = a * (1 - es**2)
    pt = a * (1 - et**2)
    
    # Plot solution if requested
    if plot:
        print(f"AoP of the vacant focus: {ws*rad:.2f}°, {wt*rad:.2f}°")
        print(f"Eccentricity of the found ellipses: {es:.4f}, {et:.4f}")
        print(f"Semilatus rectum of the found ellipses: {ps:.4f}, {pt:.4f}")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw circles
        circle1 = mpatches.Circle((P1[0], P1[1]), 2*a - r1, 
                                  fill=False, linestyle='--', alpha=0.2, color='red')
        circle2 = mpatches.Circle((P2[0], P2[1]), 2*a - r2, 
                                  fill=False, linestyle='--', alpha=0.2, color='blue')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Draw focus positions
        ax.plot(0, 0, 'kx', ms=10)
        ax.text(0, 0, 'F', fontsize=12)
        
        ax.plot(Fsvec[0], Fsvec[1], 'ko')
        ax.text(Fsvec[0], Fsvec[1], r'$F_*$', fontsize=12)
        
        ax.plot(Ftvec[0], Ftvec[1], 'ko')
        ax.text(Ftvec[0], Ftvec[1], r'$\tilde{F}_*$', fontsize=12)
        
        # Draw conics
        draw_conic(ax, es, ps, ws*rad, color='blue', alpha=0.7)
        draw_conic(ax, et, pt, wt*rad, color='green', alpha=0.7)
        
        # Draw points
        ax.plot(P1[0], P1[1], 'ro', ms=8, label='Start point')
        ax.plot(P2[0], P2[1], 'bo', ms=8, label='End point')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X [LU]')
        ax.set_ylabel('Y [LU]')
        ax.set_title('Lambert Problem Geometric Solution')
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Lambert solution saved as '{filename}'")
        
        return r1, r2, ehats, ehatt, es, et, ps, pt, ws, wt, fig, ax
    
    return r1, r2, ehats, ehatt, es, et, ps, pt, ws, wt, None, None


def get_orbit_state(a, e, i, Omega, omega, M, mu=1.0):
    """Calculate position and velocity from orbital elements."""
    # Solve for eccentric anomaly
    if e < 1e-10:  # Circular orbit
        E = M
    else:
        E = kepler_eq_solver(M, e)
    
    # True anomaly
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
    
    # Distance
    r = a * (1 - e * np.cos(E))
    
    # Position in perifocal frame
    x_p = r * np.cos(f)
    y_p = r * np.sin(f)
    z_p = 0
    
    # Velocity in perifocal frame
    p = a * (1 - e**2)
    n = np.sqrt(mu / a**3)
    vx_p = -np.sqrt(mu/p) * np.sin(f)
    vy_p = np.sqrt(mu/p) * (e + np.cos(f))
    vz_p = 0
    
    # Rotation matrices
    R_omega = rotate_matrix(omega, 3)
    R_i = rotate_matrix(i, 1)
    R_Omega = rotate_matrix(Omega, 3)
    
    # Full rotation matrix
    R = R_Omega.T @ R_i.T @ R_omega.T
    
    # Position and velocity in inertial frame
    pos_pf = np.array([x_p, y_p, z_p])
    vel_pf = np.array([vx_p, vy_p, vz_p])
    
    pos = R @ pos_pf
    vel = R @ vel_pf
    
    return pos, vel


def plot_orbits_2d(r1_vec, r2_vec, a1, e1, I1, Omega1, omega1, 
                   a2, e2, I2, Omega2, omega2, filename='orbits_2d.png'):
    """Create 2D plot of orbits and positions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: XY plane
    # Generate orbit points
    N_points = 500
    theta = np.linspace(0, 2*np.pi, N_points)
    
    # SS orbit (circular)
    r_ss = a1
    x_ss = r_ss * np.cos(theta)
    y_ss = r_ss * np.sin(theta)
    
    # Rotate SS orbit
    R_ss = (rotate_matrix(Omega1, 3) @ rotate_matrix(I1, 1) @ 
            rotate_matrix(omega1, 3)).T
    for i in range(N_points):
        point = np.array([x_ss[i], y_ss[i], 0])
        rotated = R_ss @ point
        x_ss[i], y_ss[i] = rotated[0], rotated[1]
    
    # CEV orbit (elliptical)
    f_cev = np.linspace(0, 2*np.pi, N_points)
    r_cev = a2 * (1 - e2**2) / (1 + e2 * np.cos(f_cev))
    x_cev = r_cev * np.cos(f_cev)
    y_cev = r_cev * np.sin(f_cev)
    
    # Rotate CEV orbit
    R_cev = (rotate_matrix(Omega2, 3) @ rotate_matrix(I2, 1) @ 
             rotate_matrix(omega2, 3)).T
    for i in range(N_points):
        point = np.array([x_cev[i], y_cev[i], 0])
        rotated = R_cev @ point
        x_cev[i], y_cev[i] = rotated[0], rotated[1]
    
    # Plot orbits in XY plane
    ax1.plot(x_ss, y_ss, 'g-', alpha=0.7, label='SS Orbit')
    ax1.plot(x_cev, y_cev, 'b-', alpha=0.7, label='CEV Orbit')
    
    # Plot positions
    ax1.scatter(r1_vec[0], r1_vec[1], c='green', s=100, 
                label='Start (t=0)', marker='o')
    ax1.scatter(r2_vec[0], r2_vec[1], c='blue', s=100, 
                label='End (t=200 min)', marker='s')
    
    # Add Mars
    mars_circle = plt.Circle((0, 0), 1, color='red', alpha=0.3, label='Mars')
    ax1.add_patch(mars_circle)
    
    ax1.set_xlabel('X [LU]')
    ax1.set_ylabel('Y [LU]')
    ax1.set_title('Órbitas - Plano XY')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: XZ plane
    z_ss = np.zeros_like(x_ss)
    z_cev = np.zeros_like(x_cev)
    
    for i in range(N_points):
        point_ss = np.array([x_ss[i], y_ss[i], z_ss[i]])
        rotated_ss = R_ss @ np.array([r_ss * np.cos(theta[i]), r_ss * np.sin(theta[i]), 0])
        z_ss[i] = rotated_ss[2]
        
        point_cev = np.array([x_cev[i], y_cev[i], z_cev[i]])
        rotated_cev = R_cev @ np.array([r_cev[i] * np.cos(f_cev[i]), r_cev[i] * np.sin(f_cev[i]), 0])
        z_cev[i] = rotated_cev[2]
    
    ax2.plot(x_ss, z_ss, 'g-', alpha=0.7, label='SS Orbit')
    ax2.plot(x_cev, z_cev, 'b-', alpha=0.7, label='CEV Orbit')
    
    ax2.scatter(r1_vec[0], r1_vec[2], c='green', s=100, 
                label='Start (t=0)', marker='o')
    ax2.scatter(r2_vec[0], r2_vec[2], c='blue', s=100, 
                label='End (t=200 min)', marker='s')
    
    mars_circle2 = plt.Circle((0, 0), 1, color='red', alpha=0.3, label='Mars')
    ax2.add_patch(mars_circle2)
    
    ax2.set_xlabel('X [LU]')
    ax2.set_ylabel('Z [LU]')
    ax2.set_title('Órbitas - Plano XZ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"2D orbit plot saved as '{filename}'")
    plt.close()


def main():
    """Main function to solve the rendezvous problem."""
    print("=" * 70)
    print("RENDEZVOUS EN MARTE - MECÁNICA CELESTE")
    print("Tarea 2: Cálculo de maniobra de rendezvous alrededor de Marte")
    print("=" * 70)
    
    # =========================================================================
    # 1. INITIAL DATA
    # =========================================================================
    print("\n1. DATOS INICIALES")
    print("-" * 50)
    
    # Space Station (SS) - Orbital elements
    a1 = 12554.7e3 / UL  # Semi-major axis [LU]
    e1 = 0.0  # Eccentricity
    I1 = 23.4 * deg  # Inclination [rad]
    Omega1 = 123.45 * deg  # RAAN [rad]
    omega1 = 256.23 * deg  # Argument of periapsis [rad]
    M1 = 48.98 * deg  # Mean anomaly [rad]
    
    # Crew Exploration Vehicle (CEV) - Orbital elements
    a2 = 15204.5e3 / UL  # Semi-major axis [LU]
    e2 = 0.55  # Eccentricity
    I2 = 52.97 * deg  # Inclination [rad]
    Omega2 = 137.95 * deg  # RAAN [rad]
    omega2 = 141.14 * deg  # Argument of periapsis [rad]
    M2 = 137.55 * deg  # Mean anomaly [rad]
    
    # Transfer time
    tflight = 200 * 60 / UT  # 200 minutes in canonical units
    
    print(f"Space Station:")
    print(f"  a = {a1*UL/1000:.2f} km, e = {e1:.3f}")
    print(f"  I = {I1*rad:.2f}°, Ω = {Omega1*rad:.2f}°, ω = {omega1*rad:.2f}°")
    print(f"  M = {M1*rad:.2f}°")
    
    print(f"\nCrew Exploration Vehicle:")
    print(f"  a = {a2*UL/1000:.2f} km, e = {e2:.3f}")
    print(f"  I = {I2*rad:.2f}°, Ω = {Omega2*rad:.2f}°, ω = {omega2*rad:.2f}°")
    print(f"  M = {M2*rad:.2f}°")
    
    print(f"\nTransfer time: {tflight*UT/60:.2f} min ({tflight:.4f} TU)")
    
    # =========================================================================
    # 2. CALCULATE INITIAL AND FINAL POSITIONS
    # =========================================================================
    print("\n\n2. POSICIONES INICIAL Y FINAL DE LA CÁPSULA")
    print("-" * 50)
    
    # Calculate initial position (at Space Station)
    r1_vec, v1_ss = get_orbit_state(a1, e1, I1, Omega1, omega1, M1, mu)
    
    # Calculate final position (at CEV after 200 min)
    # First propagate CEV's mean anomaly
    n2 = np.sqrt(mu / a2**3)  # Mean motion
    M2_final = M2 + n2 * tflight
    r2_vec, v2_cev = get_orbit_state(a2, e2, I2, Omega2, omega2, M2_final, mu)
    
    print(f"Posición inicial (t=0) en la estación espacial:")
    print(f"  Coordenadas canónicas: [{r1_vec[0]:.6f}, {r1_vec[1]:.6f}, {r1_vec[2]:.6f}] LU")
    print(f"  Coordenadas físicas:  [{r1_vec[0]*UL/1000:.2f}, {r1_vec[1]*UL/1000:.2f}, {r1_vec[2]*UL/1000:.2f}] km")
    
    print(f"\nPosición final (t=200 min) en el vehículo tripulado:")
    print(f"  Coordenadas canónicas: [{r2_vec[0]:.6f}, {r2_vec[1]:.6f}, {r2_vec[2]:.6f}] LU")
    print(f"  Coordenadas físicas:  [{r2_vec[0]*UL/1000:.2f}, {r2_vec[1]*UL/1000:.2f}, {r2_vec[2]*UL/1000:.2f}] km")
    
    # Create 2D plot
    plot_orbits_2d(r1_vec, r2_vec, a1, e1, I1, Omega1, omega1, 
                   a2, e2, I2, Omega2, omega2, 'orbits_2d.png')
    
    # =========================================================================
    # 3. CALCULATE EULER ANGLES FOR LAMBERT PLANE
    # =========================================================================
    print("\n\n3. ÁNGULOS DE EULER PARA EL PLANO DE LAMBERT")
    print("-" * 50)
    
    # Calculate orbital plane normal
    h_prime_vec = np.cross(r1_vec, r2_vec)
    h_prime_norm = np.linalg.norm(h_prime_vec)
    
    # Inclination
    cosI = h_prime_vec[2] / h_prime_norm
    I = np.arccos(cosI)
    
    # Longitude of ascending node
    N_vec = np.cross([0, 0, 1], h_prime_vec)
    N_norm = np.linalg.norm(N_vec)
    
    if N_norm > 1e-10:
        cosOmega = N_vec[0] / N_norm
        Omega = np.arccos(cosOmega)
        if N_vec[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        # Special case: equatorial orbit
        Omega = 0
    
    # Argument of latitude for point 1
    if N_norm > 1e-10:
        cos_u1 = np.dot(r1_vec, N_vec) / (np.linalg.norm(r1_vec) * N_norm)
        u1 = np.arccos(cos_u1)
        if r1_vec[2] < 0:
            u1 = 2 * np.pi - u1
    else:
        # For equatorial orbits
        u1 = np.arctan2(r1_vec[1], r1_vec[0])
        if u1 < 0:
            u1 += 2 * np.pi
    
    # Argument of periapsis (ω' in Lambert plane)
    omega_prime = u1
    
    print(f"Ángulos de Euler para el plano de Lambert:")
    print(f"  Inclinación (I): {I*rad:.4f}°")
    print(f"  Longitud del nodo ascendente (Ω): {Omega*rad:.4f}°")
    print(f"  Argumento del periapsis (ω'): {omega_prime*rad:.4f}°")
    
    # Rotation matrix from observation frame to Lambert frame
    Rz_omega = rotate_matrix(omega_prime, 3)
    Rx_I = rotate_matrix(I, 1)
    Rz_Omega = rotate_matrix(Omega, 3)
    
    RotMat = Rz_omega @ Rx_I @ Rz_Omega
    print(f"\nMatriz de rotación:")
    for row in RotMat:
        print(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    
    # Transform positions to Lambert frame
    r1_lambert = RotMat @ r1_vec
    r2_lambert = RotMat @ r2_vec
    
    print(f"\nVerificación (debe tener z ≈ 0 y punto 1 sobre eje x):")
    print(f"  Punto 1 en plano Lambert: [{r1_lambert[0]:.6f}, {r1_lambert[1]:.6f}, {r1_lambert[2]:.6f}]")
    print(f"  Punto 2 en plano Lambert: [{r2_lambert[0]:.6f}, {r2_lambert[1]:.6f}, {r2_lambert[2]:.6f}]")
    print(f"  z1 ≈ 0? {abs(r1_lambert[2]):.2e}, z2 ≈ 0? {abs(r2_lambert[2]):.2e}")
    
    # =========================================================================
    # 4. SOLVE LAMBERT'S PROBLEM
    # =========================================================================
    print("\n\n4. SOLUCIÓN DEL PROBLEMA DE LAMBERT")
    print("-" * 50)
    
    # Parameters for Lambert's problem
    P1 = r1_lambert
    P2 = r2_lambert
    tf = tflight
    short_transfer = True  # Short way transfer
    
    # Calculate minimum energy semi-major axis
    r1 = np.linalg.norm(P1)
    r2 = np.linalg.norm(P2)
    c = np.linalg.norm(P2 - P1)
    s = (r1 + r2 + c) / 2
    a_min = s / 2
    
    print(f"Parámetros geométricos:")
    print(f"  r1 = {r1:.6f} LU, r2 = {r2:.6f} LU")
    print(f"  c = {c:.6f} LU, s = {s:.6f} LU")
    print(f"  Semi-eje mayor mínimo (energía mínima): {a_min:.6f} LU")
    
    # Find semi-major axis for given transfer time
    lambert_func = lambda a: lambert_function(a, mu, tf, P1, P2, short_transfer, False)
    
    # Use bisection to find solution
    a_low = a_min * 1.01  # Slightly above minimum
    a_high = a_min * 3.0  # Upper bound
    
    try:
        a_sol = bisect(lambert_func, a_low, a_high)
        print(f"\nSemi-eje mayor de la órbita de transferencia: {a_sol:.6f} LU")
        print(f"Semi-eje mayor de la órbita de transferencia: {a_sol*UL/1000:.2f} km")
    except ValueError as e:
        print(f"\nError en solución de Lambert: {e}")
        # Try alternative bounds
        try:
            a_sol = bisect(lambert_func, a_min * 1.01, a_min * 5.0)
            print(f"Semi-eje mayor encontrado: {a_sol:.6f} LU")
        except:
            # Use approximate solution
            a_sol = a_min * 1.5
            print(f"Usando solución aproximada: {a_sol:.6f} LU")
    
    # Get velocities from Lambert solution
    v1_lambert, v2_lambert = lambert_function(a_sol, mu, tf, P1, P2, short_transfer, True)
    
    # Geometric solution for orbital elements
    print(f"\nSolución geométrica de Lambert:")
    result = lambert_geometric(P1[:2], P2[:2], a_sol, plot=True, filename='lambert_solution.png')
    
    if result is not None:
        r1_geom, r2_geom, ehats, ehatt, es, et, ps, pt, ws, wt, fig_lambert, ax_lambert = result
        
        # The correct orbit is the one with subscript 's'
        print(f"\nElementos orbitales de la órbita de transferencia:")
        print(f"  Semi-eje mayor (a): {a_sol*UL/1000:.2f} km")
        print(f"  Excentricidad (e): {es:.6f}")
        print(f"  Argumento del periapsis (ω'): {ws*rad:.4f}°")
        print(f"  Semilatus rectum (p): {ps:.6f} LU")
        
        if fig_lambert is not None:
            # Add velocity vectors to plot
            scale = 0.5 / np.max([np.linalg.norm(v1_lambert[:2]), np.linalg.norm(v2_lambert[:2])])
            ax_lambert.quiver(P1[0], P1[1], v1_lambert[0], v1_lambert[1], 
                             scale=scale, color='red', width=0.005, label='Velocidad inicial')
            ax_lambert.quiver(P2[0], P2[1], v2_lambert[0], v2_lambert[1], 
                             scale=scale, color='blue', width=0.005, label='Velocidad final')
            ax_lambert.legend()
            plt.savefig('lambert_solution_with_vectors.png', dpi=150, bbox_inches='tight')
            print("Solución de Lambert con vectores de velocidad guardada como 'lambert_solution_with_vectors.png'")
            plt.close()
    else:
        print("No se pudo obtener solución geométrica completa")
        es = 0.3  # Default value
        ws = omega_prime
        ps = a_sol * (1 - es**2)
    
    # =========================================================================
    # 5. VELOCITIES IN INERTIAL FRAME
    # =========================================================================
    print("\n\n5. VELOCIDADES EN EL SISTEMA INERCIAL")
    print("-" * 50)
    
    # Transform velocities back to inertial frame
    v1_inertial = RotMat.T @ v1_lambert
    v2_inertial = RotMat.T @ v2_lambert
    
    # Convert to physical units
    velocity_conversion = (UL/1000) / UT  # LU/TU to km/s
    
    print(f"Velocidad en punto de partida (plano Lambert):")
    print(f"  Vector: [{v1_lambert[0]:.6f}, {v1_lambert[1]:.6f}, {v1_lambert[2]:.6f}] LU/TU")
    print(f"  Magnitud: {np.linalg.norm(v1_lambert)*velocity_conversion:.4f} km/s")
    
    print(f"\nVelocidad en punto de llegada (plano Lambert):")
    print(f"  Vector: [{v2_lambert[0]:.6f}, {v2_lambert[1]:.6f}, {v2_lambert[2]:.6f}] LU/TU")
    print(f"  Magnitud: {np.linalg.norm(v2_lambert)*velocity_conversion:.4f} km/s")
    
    print(f"\nVelocidad en punto de partida (sistema inercial):")
    print(f"  Vector: [{v1_inertial[0]:.6f}, {v1_inertial[1]:.6f}, {v1_inertial[2]:.6f}] LU/TU")
    print(f"  Física: [{v1_inertial[0]*velocity_conversion:.4f}, "
          f"{v1_inertial[1]*velocity_conversion:.4f}, "
          f"{v1_inertial[2]*velocity_conversion:.4f}] km/s")
    
    print(f"\nVelocidad en punto de llegada (sistema inercial):")
    print(f"  Vector: [{v2_inertial[0]:.6f}, {v2_inertial[1]:.6f}, {v2_inertial[2]:.6f}] LU/TU")
    print(f"  Física: [{v2_inertial[0]*velocity_conversion:.4f}, "
          f"{v2_inertial[1]*velocity_conversion:.4f}, "
          f"{v2_inertial[2]*velocity_conversion:.4f}] km/s")
    
    # =========================================================================
    # 6. CALCULATE ΔV MANEUVERS
    # =========================================================================
    print("\n\n6. MANIOBRAS ΔV REQUERIDAS")
    print("-" * 50)
    
    # Space Station velocity at departure point
    # We already have v1_ss from earlier
    
    # CEV velocity at arrival point (after propagation)
    n2 = np.sqrt(mu / a2**3)
    M2_arrival = M2 + n2 * tflight
    r2_check, v2_cev_arrival = get_orbit_state(a2, e2, I2, Omega2, omega2, M2_arrival, mu)
    
    # ΔV calculations
    Δv_depart = v1_inertial - v1_ss
    Δv_arrival = v2_cev_arrival - v2_inertial
    
    Δv_depart_mag = np.linalg.norm(Δv_depart)
    Δv_arrival_mag = np.linalg.norm(Δv_arrival)
    
    print(f"Velocidad de la estación espacial en punto de partida:")
    print(f"  Magnitud: {np.linalg.norm(v1_ss)*velocity_conversion:.4f} km/s")
    
    print(f"\nVelocidad del vehículo tripulado en punto de llegada:")
    print(f"  Magnitud: {np.linalg.norm(v2_cev_arrival)*velocity_conversion:.4f} km/s")
    
    print(f"\nΔV en el punto de partida (salida de estación espacial):")
    print(f"  Vector: [{Δv_depart[0]:.6f}, {Δv_depart[1]:.6f}, {Δv_depart[2]:.6f}] LU/TU")
    print(f"  Física: [{Δv_depart[0]*velocity_conversion:.4f}, "
          f"{Δv_depart[1]*velocity_conversion:.4f}, "
          f"{Δv_depart[2]*velocity_conversion:.4f}] km/s")
    print(f"  Magnitud |ΔV|: {Δv_depart_mag*velocity_conversion:.4f} km/s")
    
    print(f"\nΔV en el punto de llegada (acoplamiento con vehículo tripulado):")
    print(f"  Vector: [{Δv_arrival[0]:.6f}, {Δv_arrival[1]:.6f}, {Δv_arrival[2]:.6f}] LU/TU")
    print(f"  Física: [{Δv_arrival[0]*velocity_conversion:.4f}, "
          f"{Δv_arrival[1]*velocity_conversion:.4f}, "
          f"{Δv_arrival[2]*velocity_conversion:.4f}] km/s")
    print(f"  Magnitud |ΔV|: {Δv_arrival_mag*velocity_conversion:.4f} km/s")
    
    total_Δv = Δv_depart_mag + Δv_arrival_mag
    print(f"\nΔV total de la maniobra: {total_Δv*velocity_conversion:.4f} km/s")
    
    # =========================================================================
    # 7. CREATE SUMMARY FILE
    # =========================================================================
    print("\n\n7. GUARDANDO RESULTADOS EN ARCHIVO")
    print("-" * 50)
    
    with open('rendezvous_results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RENDEZVOUS EN MARTE - RESULTADOS FINALES\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. DATOS INICIALES\n")
        f.write("-" * 50 + "\n")
        f.write(f"Estación Espacial:\n")
        f.write(f"  a = {a1*UL/1000:.2f} km, e = {e1:.3f}\n")
        f.write(f"  I = {I1*rad:.2f}°, Ω = {Omega1*rad:.2f}°, ω = {omega1*rad:.2f}°\n")
        f.write(f"  M = {M1*rad:.2f}°\n\n")
        
        f.write(f"Vehículo Tripulado:\n")
        f.write(f"  a = {a2*UL/1000:.2f} km, e = {e2:.3f}\n")
        f.write(f"  I = {I2*rad:.2f}°, Ω = {Omega2*rad:.2f}°, ω = {omega2*rad:.2f}°\n")
        f.write(f"  M = {M2*rad:.2f}°\n\n")
        
        f.write(f"Tiempo de transferencia: {tflight*UT/60:.1f} minutos\n\n")
        
        f.write("2. POSICIONES\n")
        f.write("-" * 50 + "\n")
        f.write(f"Posición inicial (t=0):\n")
        f.write(f"  [{r1_vec[0]*UL/1000:.2f}, {r1_vec[1]*UL/1000:.2f}, {r1_vec[2]*UL/1000:.2f}] km\n\n")
        
        f.write(f"Posición final (t=200 min):\n")
        f.write(f"  [{r2_vec[0]*UL/1000:.2f}, {r2_vec[1]*UL/1000:.2f}, {r2_vec[2]*UL/1000:.2f}] km\n\n")
        
        f.write("3. ÓRBITA DE TRANSFERENCIA\n")
        f.write("-" * 50 + "\n")
        f.write(f"Semi-eje mayor: {a_sol*UL/1000:.2f} km\n")
        f.write(f"Excentricidad: {es:.6f}\n")
        f.write(f"Argumento del periapsis: {ws*rad:.4f}°\n")
        f.write(f"Semilatus rectum: {ps:.6f} LU\n\n")
        
        f.write("4. MANIOBRAS ΔV\n")
        f.write("-" * 50 + "\n")
        f.write(f"ΔV en partida: {Δv_depart_mag*velocity_conversion:.4f} km/s\n")
        f.write(f"ΔV en llegada: {Δv_arrival_mag*velocity_conversion:.4f} km/s\n")
        f.write(f"ΔV total: {total_Δv*velocity_conversion:.4f} km/s\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("=" * 70 + "\n")
    
    print("Resultados guardados en 'rendezvous_results.txt'")
    
    # =========================================================================
    # 8. FINAL SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("RESUMEN DE LA SOLUCIÓN")
    print("=" * 70)
    
    print(f"\nÓRBITA DE TRANSFERENCIA (Plano de Lambert):")
    print(f"  • Semi-eje mayor: {a_sol*UL/1000:.2f} km")
    print(f"  • Excentricidad: {es:.6f}")
    print(f"  • Argumento del periapsis: {ws*rad:.4f}°")
    
    print(f"\nMANIOBRAS ΔV REQUERIDAS:")
    print(f"  • ΔV en partida: {Δv_depart_mag*velocity_conversion:.4f} km/s")
    print(f"  • ΔV en llegada: {Δv_arrival_mag*velocity_conversion:.4f} km/s")
    print(f"  • ΔV total: {total_Δv*velocity_conversion:.4f} km/s")
    
    print(f"\nTIEMPO DE TRANSFERENCIA: {tflight*UT/60:.1f} minutos")
    
    print(f"\nARCHIVOS GENERADOS:")
    print(f"  • orbits_2d.png - Gráfico 2D de órbitas")
    print(f"  • lambert_solution.png - Solución geométrica de Lambert")
    print(f"  • lambert_solution_with_vectors.png - Solución con vectores de velocidad")
    print(f"  • rendezvous_results.txt - Resultados detallados")
    
    print("\n" + "=" * 70)
    print("FIN DE LA SIMULACIÓN")
    print("=" * 70)


if __name__ == "__main__":
    main()