# from math import cos, sin, pi
# from amuse.ext.orbital_elements import (
#         new_binary_from_orbital_elements,
#         # orbital_elements_from_binary,
#         )
import numpy
from amuse.datamodel import Particles
from amuse.ext.orbital_elements1 import generate_binaries
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.io import write_set_to_file
from amuse.units import constants
from amuse.units import units
from numpy import pi
from numpy.random import random
from numpy.random import uniform


def circular_velocity(
    primary,
    secondary,
    G=constants.G,
):
    v_circ = (G * (primary.mass + secondary.mass) /
              (secondary.position - primary.position).length())**0.5
    return v_circ


def orbital_period_to_semi_major_axis(orbital_period,
                                      mass1,
                                      mass2=0 | units.MSun,
                                      G=constants.G):
    """
    returns semi-major axis for given period and masses
    """
    mu = G * (mass1 + mass2)
    a3 = (orbital_period / (2 * pi))**2 * mu
    return a3**(1.0 / 3)


def new_binary_distribution(
    primary_masses,
    secondary_masses=None,
    binaries=None,
    min_mass=0.1 | units.MSun,
):
    """
    Takes primary masses, and returns a set of stars and a set of binaries
    formed by these stars.
    Secondary masses are given by a uniform random mass ratio with the primary
    masses. If optional secondary masses are given, these are used instead.
    binaries is an optional particleset used for the binary pairs, with given
    positions and velocities. Other parameters are ignored.
    """
    number_of_primaries = len(primary_masses)
    if binaries is None:
        binaries = Particles(number_of_primaries)
        # Should give some position/velocity as well?
    elif len(binaries) != number_of_primaries:
        print("binaries must be None or have the same lenght as primary_mass")
        return -1
    if secondary_masses is None:
        # Now, we need to specify the mass ratio in the binaries.
        # A flat distribution seems to be OK.
        mass_ratio = uniform(0,1,number_of_primaries)
        #mass_ratio = uniform(size=number_of_primaries)
        # This gives us the secondaries' masses
        secondary_masses = min_mass +  ((primary_masses - min_mass) * mass_ratio)
        # secondaries are min_mass at least
        secondary_masses[secondary_masses < min_mass] = min_mass
    elif len(secondary_masses) != number_of_primaries:
        print("Number of secondaries is unequal to number of primaries!")
        return -1
    else:
        # Make sure primary_mass is the larger of the two, and secondary_mass
        # the smaller.
        pm = primary_masses.maximum(secondary_masses)
        sm = primary_masses.minimum(secondary_masses)
        primary_masses = pm
        secondary_masses = sm
        del (pm, sm)
    # Now, we need to calculate the semi-major axes for the binaries. Since the
    # observed quantity is orbital periods, we start from there.
    mean_log_orbital_period = 5  # 10log of the period in days, (Duchene&Kraus)
    sigma_log_orbital_period = 2.3
    orbital_period = (numpy.random.lognormal(
        size=number_of_primaries,
        mean=numpy.log(10) * mean_log_orbital_period,
        sigma=numpy.log(10) * sigma_log_orbital_period,
    )
                      | units.day)
    # We need the masses to calculate the corresponding semi-major axes.
#     semi_major_axis = orbital_period_to_semi_major_axis(
#         orbital_period,
#         primary_masses,
#         secondary_masses,
#     )
    
    #FLAT AU DISTR
    semi_major_axis = 1 | units.au
    
    # Eccentricity: square root of random value
    eccentricity = 0
  #  eccentricity = numpy.sqrt(random(number_of_primaries))
    # Other orbital elements at random
    inclination = pi * random(number_of_primaries) | units.rad
    true_anomaly = 2 * pi * random(number_of_primaries) | units.rad
    longitude_of_the_ascending_node = (2 * pi * random(number_of_primaries)) | units.rad
    argument_of_periapsis = (2 * pi * random(number_of_primaries)) | units.rad
    primaries, secondaries, coms, coms_v = generate_binaries(
        primary_masses,
        secondary_masses,
        semi_major_axis,
        eccentricity=eccentricity,
        inclination=inclination,
        true_anomaly=true_anomaly,
        longitude_of_the_ascending_node=longitude_of_the_ascending_node,
        argument_of_periapsis=argument_of_periapsis,
        G=constants.G,
    )
    stars = Particles()

    if hasattr(binaries, "position"):
        primaries.position += binaries.position
        secondaries.position += binaries.position
    if hasattr(binaries, "velocity"):
        primaries.velocity += binaries.velocity
        secondaries.velocity += binaries.velocity
    primaries = stars.add_particles(primaries)
    secondaries = stars.add_particles(secondaries)
   # binaries.com = coms
   # binaries.com_v = coms_v
    binaries.x = coms.x
    binaries.y = coms.y
    binaries.z = coms.z
    
    binaries.vx = coms_v.x
    binaries.vy = coms_v.y
    binaries.vz = coms_v.z
    
    binaries.inclination = inclination
    binaries.eccentricity = eccentricity
    binaries.semi_major_axis = semi_major_axis
    for i, primary in enumerate(primaries):
        binaries[i].child1 = primary
        
        binaries[i].child2 = secondaries[i]
        
        # Probably needed
        binaries[i].mass = primary.mass + secondaries[i].mass
    
    
    #need the positions and velocities and masses using .chil1/2
    binaries.child1_mass = primaries.mass
    binaries.child2_mass = secondaries.mass
    binaries.child1.x = primary.x
    binaries.child1.y = primary.y
    binaries.child1.z = primary.z
    binaries.child1.vx = primary.vx
    binaries.child1.vy = primary.vy
    binaries.child1.vz = primary.vz
    
    binaries.child2.x = secondaries.x
    binaries.child2.y = secondaries.y
    binaries.child2.z = secondaries.z
    binaries.child2.vx = secondaries.vx
    binaries.child2.vy = secondaries.vy
    binaries.child2.vz = secondaries.vz
        
    return stars, binaries


def main():
    number_of_pairs = 4
    primary_masses = new_kroupa_mass_distribution(number_of_pairs)
    stars, binaries = new_binary_distribution(primary_masses, )
    write_set_to_file(stars, "stars.amuse")
    write_set_to_file(binaries, "binaries.amuse")


if __name__ == "__main__":
    main()

