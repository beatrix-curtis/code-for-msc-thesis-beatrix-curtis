#!/usr/bin/env python
"""
MASC creates a model star cluster, which can then be used in N-body simulations
or for other purposes.

It requires AMUSE, which can be downloaded from http://amusecode.org or
https://github.com/amusecode/amuse.

Currently not feature-complete yet, and function/argument names are
subject to change.
"""

import logging
import numpy

from amuse.units import (
    units,
    nbody_system,
    generic_unit_converter,
)
from amuse.units.trigo import sin, cos
from amuse.datamodel.particles import Particles
from amuse.ic.plummer import new_plummer_sphere
from amuse.ic.kingmodel import new_king_model
from amuse.ext.masc.binaries_updated import new_binary_distribution
from numpy.random import random
from numpy.random import uniform
try:
    from amuse.ic.fractalcluster import new_fractal_cluster_model
except ImportError:
    new_fractal_cluster_model = None


def new_masses(
    stellar_mass=False,
    initial_mass_function="kroupa",
    upper_mass_limit=125. | units.MSun,
    lower_mass_limit=0.1 | units.MSun,
    exceed_mass=True,
    number_of_stars=1024,
    sort_by_mass=False,
):
    """
    Creates new stellar masses.
    """
    imf_name = initial_mass_function.lower()
    if imf_name == "salpeter":
        from amuse.ic.salpeter import new_salpeter_mass_distribution
        initial_mass_function = new_salpeter_mass_distribution
    elif imf_name == "kroupa":
        from amuse.ic.brokenimf import new_kroupa_mass_distribution
        initial_mass_function = new_kroupa_mass_distribution
    elif imf_name == "flat":
        from amuse.ic.flatimf import new_flat_mass_distribution
        initial_mass_function = new_flat_mass_distribution
    elif imf_name == "fixed":
        from amuse.ic.flatimf import new_flat_mass_distribution

        def new_fixed_mass_distribution(
                number_of_particles, *list_arguments, **keyword_arguments
        ):
            return new_flat_mass_distribution(
                number_of_particles,
                mass_min=stellar_mass/number_of_stars,
                mass_max=stellar_mass/number_of_stars,
            )
        initial_mass_function = new_fixed_mass_distribution

    if stellar_mass:
        # best underestimate mean_mass a bit for faster results
        mean_mass = max(0.25 | units.MSun, lower_mass_limit)
        mass = initial_mass_function(
            max(1, int(stellar_mass / mean_mass)),
            mass_min=lower_mass_limit,
            mass_max=upper_mass_limit,
        )
        previous_number_of_stars = len(mass)
        if exceed_mass:
            # Allow one final star to exceed stellar_mass
            final_star = 1+numpy.argmax(mass.cumsum() > stellar_mass)
            if (final_star > 1 and final_star < len(mass)):
                mass = mass[:final_star]
        else:
            # Limit to stars not exceeding stellar_mass
            mass = mass[mass.cumsum() < stellar_mass]

        additional_mass = [] | units.MSun
        while True:
            if stellar_mass < mass.sum():
                break
            if previous_number_of_stars + len(additional_mass) > len(mass):
                break
            # We don't have enough stars yet, or at least not tested this
            additional_mass = initial_mass_function(
                max(1, int(stellar_mass / mean_mass)),
                mass_min=lower_mass_limit,
                mass_max=upper_mass_limit,
            )
            if exceed_mass:
                # Allow one final star to exceed stellar_mass
                final_star = 1+numpy.argmax(
                    mass.sum() + additional_mass.cumsum() > stellar_mass
                )
                if (final_star > 1 and final_star < len(mass)):
                    additional_mass = additional_mass[:final_star]
                mass.append(additional_mass)
            else:
                # Limit to stars not exceeding stellar_mass
                additional_mass_used = additional_mass[
                    mass.sum() + additional_mass.cumsum() < stellar_mass
                ]
                mass.append(additional_mass_used)
                if len(additional_mass_used) < len(additional_mass):
                    break
        number_of_stars = len(mass)
    else:
        # Give stars their mass
        mass = initial_mass_function(
            number_of_stars,
            mass_min=lower_mass_limit,
            mass_max=upper_mass_limit,
        )

    if sort_by_mass:
        mass = mass.sorted()[::-1]
        if exceed_mass:
            final_star = 1+numpy.argmax(mass.cumsum() > stellar_mass)
            if (final_star > 1 and final_star < len(mass)):
                mass = mass[:final_star]

    return mass


def new_star_cluster(
       # stellar_mass=False,
        rand_seed,
        stellar_mass,
       # initial_mass_function="salpeter",
        initial_mass_function,
        #upper_mass_limit=125. | units.MSun,
        upper_mass_limit,
       # lower_mass_limit=0.1 | units.MSun,
        lower_mass_limit,
        #number_of_stars=1024,
        number_of_stars,
       # effective_radius=3.0 | units.parsec,
        effective_radius,
       # star_distribution="plummer",
        star_distribution,
      #  star_distribution_w0=7.0,
        star_distribution_w0,
      #  star_distribution_fd=2.0,
        star_distribution_fd,
       # star_metallicity=0.01,
        star_metallicity,
        # initial_binary_fraction=0,
        binary_frac,
        **kwargs
):
    """
    Create stars.
    When using an IMF, either the stellar mass is fixed (within
    stochastic error) or the number of stars is fixed. When using
    equal-mass stars, both are fixed.
    """
    number_of_stars = number_of_stars
    
    #single star masses
    masses = new_masses(
        stellar_mass=stellar_mass,
        initial_mass_function=initial_mass_function,
        upper_mass_limit=upper_mass_limit,
        lower_mass_limit=lower_mass_limit,
        number_of_stars=round((1 - (0.5*binary_frac))*number_of_stars),
    )
    print("len masses", len(masses))
    print("single star length", len(masses))

    if binary_frac != 0:
        #create the primary masses for the binary population, half of the binary fraction in this case
        primary_masses = new_masses(
        stellar_mass=stellar_mass,
        initial_mass_function=initial_mass_function,
        upper_mass_limit=upper_mass_limit,
        lower_mass_limit=lower_mass_limit,
        number_of_stars=round(0.5 * binary_frac * number_of_stars),
        )
        prim_total_mass = primary_masses.sum()
        #create the binary population
        binary_stars, binaries = new_binary_distribution(primary_masses)
        print("binary star number", len(binary_stars))


    converter = nbody_system.nbody_to_si(masses.sum(), effective_radius)
    # Give stars a position and velocity, based on the distribution model.
    if star_distribution == "plummer":
        stars = new_plummer_sphere(
            round((1 - (0.5*binary_frac))*number_of_stars),
            convert_nbody=converter,
        )
    elif star_distribution == "king":
        stars = new_king_model(
            round(number_of_stars* (1-binary_frac)),
            star_distribution_w0,
            convert_nbody=converter,
        )
    elif star_distribution == "fractal":
        stars = new_fractal_cluster_model(
            round(number_of_stars* (1-binary_frac)),
            fractal_dimension=star_distribution_fd,
            convert_nbody=converter,
        )
    else:
        return -1, "No stellar distribution"
    
    #array of positions to replace with binaries, shuffled so that the stars are replaced randomly
    arr = numpy.array([1] * round(0.5 *number_of_stars* binary_frac) + [0] * (round((1 - (0.5*binary_frac))*number_of_stars)-round(0.5 *number_of_stars* binary_frac)))
    numpy.random.shuffle(arr)
    arr
    #scale the binary population with the mass and effective radius
    if binary_frac != 0:
        
        converter_binaries = nbody_system.nbody_to_si(binaries.mass.sum(), effective_radius)

        binaries.move_to_center()
        binaries.scale_to_standard(
                convert_nbody=converter_binaries,
                # virial_ratio=virial_ratio,
                # smoothing_length_squared= ...,
            )
        #now I am trying to use the array of binaries to replace stars with binaries.
        a = 0
        #masses2 will be the single star masses
        masses2_list = []
        for num in arr:
            if num == 0:
                masses2_list.append(masses[a])
            a +=1
        a = 0
        b = 0
        num_binaries = 0
        #this bit replaces random single stars with the binary stars created earlier
        for num in arr:
            if num == 1:
                binaries[b].position = stars[a].position.value_in(units.pc) | units.pc
                binaries[b].velocity = stars[a].velocity.value_in(units.kms) | units.kms
                stars.remove_particle(stars[a])
                stars.add_particle(binaries[b])
                num_binaries += 1
                b+=1
                a=a-1

            a+=1
        print("binary number",num_binaries)
        #a and b should correspond to the number of single stars and binaries respectively
        print("a is ", a)
        print("b is ", b)
        print(" ")
        #array for plotting singles and binaries
        new_arr = numpy.array([1] * (round((1 - (0.5*binary_frac))*number_of_stars)-round(0.5 *number_of_stars* binary_frac)) + [0] *
                             round(0.5 *number_of_stars* binary_frac))
        #array for plotting singles and binary components
        new_arr1 = numpy.array([1] * (round((1 - (0.5*binary_frac))*number_of_stars)-round(0.5 *number_of_stars* binary_frac)) + [0] *
                             round(number_of_stars* binary_frac))
        
        masses2 = numpy.array(masses2_list)

    if binary_frac == 0:
        stars.mass = masses
        new_arr1 = 0
    else:
        stars.mass = numpy.concatenate((masses2, binaries.mass), axis=None)
        #should be equal to number_of_stars
        print("stars new length", len(stars))
    #scale the whole cluster now
    converter_all = nbody_system.nbody_to_si(stars.mass.sum(), effective_radius)
    stars.move_to_center()
    stars.scale_to_standard(
            convert_nbody=converter_all,
            # virial_ratio=virial_ratio,
            # smoothing_length_squared= ...,
        )

    stars.metallicity = star_metallicity
        


    # Record the cluster's initial parameters to the particle distribution
    stars.collection_attributes.initial_mass_function = \
        initial_mass_function.lower()
    stars.collection_attributes.upper_mass_limit = upper_mass_limit
    stars.collection_attributes.lower_mass_limit = lower_mass_limit
    stars.collection_attributes.number_of_stars = len(stars)

    stars.collection_attributes.effective_radius = effective_radius

    stars.collection_attributes.star_distribution = star_distribution
    stars.collection_attributes.star_distribution_w0 = star_distribution_w0
    stars.collection_attributes.star_distribution_fd = star_distribution_fd

    stars.collection_attributes.star_metallicity = star_metallicity
    stars.collection_attributes.binary_frac = binary_frac

    # Derived/legacy values
    stars.collection_attributes.converter_mass = \
        converter_all.to_si(1 | nbody_system.mass)
    stars.collection_attributes.converter_length =\
        converter_all.to_si(1 | nbody_system.length)
    stars.collection_attributes.converter_speed =\
        converter_all.to_si(1 | nbody_system.speed)
    if binary_frac == 0:
        binary_stars = 0
        binaries = 0

    return stars, binary_stars, binaries, new_arr1 


def new_stars_from_sink(
        origin,
        upper_mass_limit=125 | units.MSun,
        lower_mass_limit=0.1 | units.MSun,
        default_radius=0.25 | units.pc,
        velocity_dispersion=1 | units.kms,
        logger=None,
        initial_mass_function="kroupa",
        distribution="random",
        randomseed=None,
        **keyword_arguments
):
    """
    Form stars from an origin particle that keeps track of the properties of
    this region.
    """
    logger = logger or logging.getLogger(__name__)
    if randomseed is not None:
        logger.info("setting random seed to %i", randomseed)
        numpy.random.seed(randomseed)

    try:
        initialised = origin.initialised
    except AttributeError:
        initialised = False
    if not initialised:
        logger.debug(
            "Initialising origin particle %i for star formation",
            origin.key
        )
        next_mass = new_star_cluster(
            initial_mass_function=initial_mass_function,
            upper_mass_limit=upper_mass_limit,
            lower_mass_limit=lower_mass_limit,
            number_of_stars=1,
            **keyword_arguments
        )
        origin.next_primary_mass = next_mass[0].mass
        origin.initialised = True

    if origin.mass < origin.next_primary_mass:
        logger.debug(
            "Not enough in star forming region %i to form the next star",
            origin.key
        )
        return Particles()

    mass_reservoir = origin.mass - origin.next_primary_mass
    stellar_masses = new_star_cluster(
        stellar_mass=mass_reservoir,
        upper_mass_limit=upper_mass_limit,
        lower_mass_limit=lower_mass_limit,
        imf=initial_mass_function,
    ).mass
    number_of_stars = len(stellar_masses)

    new_stars = Particles(number_of_stars)
    new_stars.age = 0 | units.yr
    new_stars[0].mass = origin.next_primary_mass
    new_stars[1:].mass = stellar_masses[:-1]
    origin.next_primary_mass = stellar_masses[-1]
    new_stars.position = origin.position
    new_stars.velocity = origin.velocity

    try:
        radius = origin.radius
    except AttributeError:
        radius = default_radius
    rho = numpy.random.random(number_of_stars) * radius
    theta = (
        numpy.random.random(number_of_stars)
        * (2 * numpy.pi | units.rad)
    )
    phi = (
        numpy.random.random(number_of_stars) * numpy.pi | units.rad
    )
    x = rho * sin(phi) * cos(theta)
    y = rho * sin(phi) * sin(theta)
    z = rho * cos(phi)
    new_stars.x += x
    new_stars.y += y
    new_stars.z += z

    velocity_magnitude = numpy.random.normal(
        scale=velocity_dispersion.value_in(units.kms),
        size=number_of_stars,
    ) | units.kms
    velocity_theta = (
        numpy.random.random(number_of_stars)
        * (2 * numpy.pi | units.rad)
    )
    velocity_phi = (
        numpy.random.random(number_of_stars)
        * (numpy.pi | units.rad)
    )
    vx = velocity_magnitude * sin(velocity_phi) * cos(velocity_theta)
    vy = velocity_magnitude * sin(velocity_phi) * sin(velocity_theta)
    vz = velocity_magnitude * cos(velocity_phi)
    new_stars.vx += vx
    new_stars.vy += vy
    new_stars.vz += vz

    new_stars.origin = origin.key
    origin.mass -= new_stars.total_mass()

    return new_stars

