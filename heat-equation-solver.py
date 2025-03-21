import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters from the example
L = 1  # Length of the rod
c_squared = 0.45  # Thermal diffusivity
k = 0.01  # Temporal step size
h = 0.1  # Spatial step size
n = int(L / h)  # Number of spatial segments
s = c_squared * k / h**2  # Stability parameter

# Time and space points
time_points = np.arange(0, 1.0 + k, k)  # All time steps
x_points = np.arange(0, L + h, h)  # Discrete spatial points
specified_time_points = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Specified time points

# Initialize the temperature array
u = np.zeros((len(x_points), len(time_points)))

# Boundary conditions: u(0, t) = u(L, t) = 0
u[0, :] = 0
u[-1, :] = 0

# Initial condition: u(x, 0) = 100x
u[:, 0] = 100 * x_points

# Finite difference scheme
for j in range(1, len(time_points)):
    for i in range(1, n):
        u[i, j] = (1 - 2 * s) * u[i, j-1] + s * (u[i+1, j-1] + u[i-1, j-1])

# Extract solutions at specified time points
specified_indices = [int(t / k) for t in specified_time_points]
u_specified = u[:, specified_indices]

# Store the approximate solutions in a DataFrame
approx_solutions = pd.DataFrame(u_specified, index=x_points, columns=specified_time_points)

# Define the analytical solution for the heat equation
def analytical_solution(x, t, L, c_squared):
    solution = np.zeros_like(x)
    for m in range(1, 10000):  # Use the first 10000 terms of the series
        term = ((-1) ** (m + 1)) * (200 / (m * np.pi)) * np.sin(m * np.pi * x / L) * np.exp(-c_squared * (m * np.pi / L)**2 * t)
        solution += term
    return solution

# Calculate the exact solutions at the specified discrete points
exact_solutions = np.zeros_like(u_specified)
for j, t in enumerate(specified_time_points):
    exact_solutions[:, j] = analytical_solution(x_points, t, L, c_squared)

# Initial condition: u(x, 0) = 100x
exact_solutions[:, 0] = 100 * x_points

# Store the exact solutions in a DataFrame
exact_solutions_df = pd.DataFrame(exact_solutions, index=x_points, columns=specified_time_points)

# Round the values in the tables
approx_solutions = approx_solutions.round(2)
exact_solutions_df = exact_solutions_df.round(2)

# Transpose the approximate solutions table
approx_solutions_transposed = approx_solutions.T

# Create an image output of the transposed table for approximate solutions
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

col_labels = [f"x = {col:.1f}" for col in approx_solutions_transposed.columns]
row_labels = [f"t = {row}" for row in approx_solutions_transposed.index]

transposed_table = ax.table(cellText=approx_solutions_transposed.values, colLabels=col_labels, rowLabels=row_labels, loc='center', cellLoc='center')
transposed_table.auto_set_font_size(False)
transposed_table.set_fontsize(10)
transposed_table.auto_set_column_width(col=list(range(len(approx_solutions_transposed.columns))))

plt.title("Approximate Solutions at Discrete Points")
plt.savefig("transposed_approximate_solutions_table.png")
plt.close()

# Transpose the exact solutions table
exact_solutions_transposed = exact_solutions_df.T

# Create an image output of the transposed table for exact solutions
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

col_labels = [f"x = {col:.1f}" for col in exact_solutions_transposed.columns]
row_labels = [f"t = {row}" for row in exact_solutions_transposed.index]

transposed_table_exact = ax.table(cellText=exact_solutions_transposed.values, colLabels=col_labels, rowLabels=row_labels, loc='center', cellLoc='center')
transposed_table_exact.auto_set_font_size(False)
transposed_table_exact.set_fontsize(10)
transposed_table_exact.auto_set_column_width(col=list(range(len(exact_solutions_transposed.columns))))

plt.title("Exact Solutions at Discrete Points")
plt.savefig("transposed_exact_solutions_table.png")
plt.close()

# Plot the approximate solution against the exact solution for t = 0.6
t_index = 3  # Index for t = 0.6 in specified_time_points

# Extract data for plotting
x_values = x_points
approx_values = approx_solutions.iloc[:, t_index]

# Generate more points for a smoother exact solution curve
x_fine = np.linspace(0, L, 1000)
exact_values_fine = analytical_solution(x_fine, 0.6, L, c_squared)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_fine, exact_values_fine, label="Exact Solution", color="blue")
plt.plot(x_values, approx_values, label="Approximate Solutions", color="red", marker="o", linestyle="-")

plt.title("Approximate vs Exact Solutions at t = 0.6")
plt.xlabel("Position x")
plt.ylabel("Temperature u(x, t)")
plt.legend()
plt.grid(True)
plt.savefig("approx_vs_exact_t0.6.png")
plt.show()
