# https://hockygroup.hosting.nyu.edu/exercise/langevin-dynamics.html
import numpy as np
import pandas as pd
import random

# this is step A
def position_update(x, v, dt):
    x_new = x + v * dt / 2.
    return x_new

# this is step B
def velocity_update(v, F, dt):
    v_new = v + F * dt / 2.
    return v_new

def random_velocity_update(v, gamma, kBT, dt):
    R = np.random.normal()
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(1 - c1 * c1) * np.sqrt(kBT)
    v_new = c1 * v + R * c2
    return v_new

def baoab(potential, max_steps, dt, gamma, kBT, initial_position, initial_velocity,
          save_frequency=100, **kwargs):
    """
    provide the initial conditions and the potential function, and this function will run Langevin dynamics
    :param potential: a function that takes a position and returns the potential_energy and force
    :param max_steps:
    :param dt:
    :param gamma:
    :param kBT:
    :param initial_position:
    :param initial_velocity:
    :param save_frequency:
    :param kwargs:
    :return: positions, (x, v)
        positions: a numpy array of trajectory
        (x, v): the final position and velocity
    """
    x = initial_position
    v = initial_velocity
    t = 0
    step_number = 0
    positions = []
    # velocities = []
    # potential_energies = []
    # save_times = []

    while (step_number <= max_steps):

        # B
        potential_E, force = potential(x, **kwargs)
        v = velocity_update(v, force, dt)

        # A
        x = position_update(x, v, dt)

        # O
        v = random_velocity_update(v, gamma, kBT, dt)

        # A
        x = position_update(x, v, dt)

        # B
        potential_E, force = potential(x, **kwargs)
        v = velocity_update(v, force, dt)

        if step_number % save_frequency == 0 and step_number > 0:
            kinetic = .5 * v * v
            e_total = kinetic + potential_E

            positions.append(x)
            # velocities.append(v)
            # potential_energies.append(potential_E)
            # save_times.append(t)

        t = t + dt
        step_number = step_number + 1

    positions = np.array(positions, dtype=np.single)

    return positions, (x, v)
