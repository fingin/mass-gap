import numpy as np
import random

# Define parameters
N = 15  # lattice size
beta = 10  # coupling constant
nsteps = 100000  # number of simulation steps

# initialize links with random SU(2) matrices
links = np.zeros((N, N, N, 4), dtype=np.ndarray)
for x in range(N):
    for y in range(N):
        for z in range(N):
            for mu in range(4):
                links[x,y,z,mu] = np.identity(2) + 0.1 * (np.random.rand(2,2) - 0.5)

# Define the plaquette action
def plaquette(x, y, z, mu, nu):
    P = np.identity(2)
    for i in range(mu, nu):
        P = np.dot(P, links[x,y,z,i])
    for i in range(mu):
        P = np.dot(P, np.conj(np.transpose(links[x,y,z,i])))
    for i in range(nu, mu, -1):
        P = np.dot(P, np.conj(np.transpose(links[(x+mu)%N,(y+nu)%N,(z+nu)%N,i])))
    for i in range(nu, mu, -1):
        P = np.dot(P, np.transpose(links[x,y,z,i]))
    return np.trace(P).real

# Define the action
def action():
    S = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for mu in range(4):
                    for nu in range(mu):
                        S += beta * (1 - plaquette(x, y, z, mu, nu))
    return S

# Perform the simulation
for step in range(nsteps):
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for mu in range(4):
                    U = links[x,y,z,mu]
                    dS = 0
                    for nu in range(4):
                        if nu != mu:
                            dS += beta * (plaquette(x, y, z, mu, nu) - plaquette(x, y, z, nu, mu) - plaquette((x+mu)%N, (y+nu)%N, (z+nu)%N, nu, mu) + plaquette((x+nu)%N, (y+nu)%N, (z+nu)%N, mu, nu))
                    U_new = np.dot(np.exp(-dS), U)
                    links[x,y,z,mu] = U_new / np.sqrt(np.trace(np.dot(np.conj(np.transpose(U_new)), U_new)).real)

    if step % 1 == 0:
        print("Step %d, action %f" % (step, action()))
