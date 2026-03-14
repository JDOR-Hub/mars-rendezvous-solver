# Mars Rendezvous – Celestial Mechanics
This Python project solves a rendezvous problem around Mars using orbital mechanics. It calculates the required transfer orbit and ΔV maneuvers for a spacecraft to meet another vehicle after a specified time, given their initial orbital elements.

The script implements:
- Conversion of orbital elements to Cartesian state vectors.
- Propagation of mean anomaly.
- Determination of the Lambert transfer plane via Euler angles.
- Solution of Lambert's problem (short-way transfer) using geometric and numerical methods.
- Computation of velocity changes at departure and arrival.
- Generation of 2D plots of the orbits and the Lambert solution.

All quantities are expressed in canonical units (LU = Mars radius, TU = derived time unit, mu = 1) for numerical stability, then converted to physical units (km, km/s).

## Features
- Orbital state calculation from Keplerian elements (a, e, i, Ω, ω, M).
- Lambert's problem solver (bisection + geometric construction) to find the transfer orbit.
- Euler angles to transform the problem into the transfer plane.
- ΔV computation for both departure and arrival.

### Plot generation:
* [orbits_2d.png](orbits_2d.png) – XY and XZ projections of the initial orbits.
* [lambert_solution.png](lambert_solution.png) – Geometric construction of Lambert's problem (circles and ellipses).
* [lambert_solution_with_vectors.png](lambert_solution_with_vectors.png) – Same with velocity vectors.
* Text output ([rendezvous_results.txt](rendezvous_results.txt)) with full numerical results.
