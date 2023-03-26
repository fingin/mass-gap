import numpy as np

# Define parameters
N = 30  # lattice size
beta = 2.2  # coupling constant
nsteps = 100000  # number of simulation steps

# Initialize lattice
lattice = np.random.choice([-1, 1], size=(N, N))

# Define energy calculation function
def calculate_energy(lattice, beta):
    energy = 0
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            spin = lattice[i][j]
            nb_sum = lattice[(i+1)%N][j] + lattice[i][(j+1)%N] + lattice[(i-1)%N][j] + lattice[i][(j-1)%N]
            energy += -nb_sum*spin
    energy *= beta
    return energy

# Define function to run simulation
def simulate(lattice, beta, nsteps):
    action_values = []
    energy_values = []
    for step in range(nsteps):
        # Choose random spin and flip it
        i = np.random.randint(N)
        j = np.random.randint(N)
        lattice[i][j] *= -1

        # Calculate change in energy and accept or reject move based on Metropolis algorithm
        energy_before = calculate_energy(lattice, beta)
        lattice[i][j] *= -1
        energy_after = calculate_energy(lattice, beta)
        delta_E = energy_after - energy_before
        action = np.exp(-delta_E)
        if np.random.rand() < action:
            lattice[i][j] *= -1
            energy_before = energy_after
        
        # Record action and energy at every step
        action_values.append(action)
        energy_values.append(energy_before)

        # Print progress every 10% of the way through simulation
        if (step+1) % (nsteps//10) == 0:
            print(f"Completed {step+1} steps out of {nsteps}")

    return lattice, action_values, energy_values

# Run simulation
final_lattice, action_values, energy_values = simulate(lattice, beta, nsteps)

# Calculate mass gap
correlation_values = []
for i in range(N):
    for j in range(N):
        spin = final_lattice[i][j]
        nb_sum = final_lattice[(i+1)%N][j] + final_lattice[i][(j+1)%N] + final_lattice[(i-1)%N][j] + final_lattice[i][(j-1)%N]
        correlation_values.append(spin*nb_sum)
mass_gap = np.abs(np.sum(correlation_values))/N**2
print(f"Mass gap: {mass_gap}")
