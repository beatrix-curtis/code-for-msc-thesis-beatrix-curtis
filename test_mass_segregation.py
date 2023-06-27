from amuse.units import nbody_system, units, constants
from amuse.ic.plummer import new_plummer_model
from amuse.community.ph4 import Ph4
from amuse.community.smalln import Smalln
from amuse.community.kepler import Kepler
from amuse.couple import multiples
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.ic.flatimf import new_flat_mass_distribution
from amuse.ic.salpeter import new_powerlaw_mass_distribution_nbody
from amuse.datamodel import Particles
from amuse.datamodel import particle_attributes
from amuse.datamodel import particles
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import random
import numpy as np
import pickle
from amuse.ext.orbital_elements import get_orbital_elements_from_binaries, generate_binaries

#from amuse.ext.mas.cluster_binaries_random_mass import new_star_cluster new_star_cluster, new_masses
from amuse.ext.masc.cluster_binaries_high_masses_in_binaries import new_star_cluster, new_masses
from amuse.ext.masc.binaries_updated import new_binary_distribution

# Fix random seed for reproducibility
num = random.randint(1,1000000)
np.random.seed(num)

print("random number", num)

class GravityWithMultiples:
    def __init__(self, stars,

              #   initial_binaries_stars,

                 converter=None, time_step=0.01 | units.Myr):
        self.time_step = time_step
        self.mass_total = stars.total_mass()
        self.number_of_particles = len(stars)
        if converter is None:
            self.converter = nbody_system.nbody_to_si(
                self.mass_total,
                self.time_step
            )
        else:
            self.converter = converter
        self.model_time = 0 * self.time_step

        self.SmallN = None
        self.gravity = Ph4(convert_nbody=self.converter)
        self.gravity.parameters.epsilon_squared = (self.converter.to_si(0.0 | nbody_system.length))**2
        self.gravity.particles.add_particles(stars)
        #TEMP ADD BINARIES
     #   self.gravity.particles.add_particles(initial_binaries_stars)

        stopping_condition = self.gravity.stopping_conditions.collision_detection
        stopping_condition.enable()

        self.init_smalln()
        self.kep = Kepler(unit_converter=self.converter)
        self.kep.initialize_code()
        self.multiples_code = multiples.Multiples(
            self.gravity, self.new_smalln, self.kep, constants.G
        )
        self.multiples_code.neighbor_perturbation_limit = 0.05
        self.multiples_code.global_debug = 1

        #  global_debug = 0: no output from multiples
        #                 1: minimal output
        #                 2: debugging output
        #                 3: even more output

        print('')
        print(f'multiples_code.neighbor_veto = {self.multiples_code.neighbor_veto}')
        print(f'multiples_code.neighbor_perturbation_limit = {self.multiples_code.neighbor_perturbation_limit}')
        print(f'multiples_code.retain_binary_apocenter = {self.multiples_code.retain_binary_apocenter}')
        print(f'multiples_code.wide_perturbation_limit = {self.multiples_code.wide_perturbation_limit}')

        self.energy_0 = self.print_diagnostics(self.multiples_code)
        j=num
        with open(f'log_test/log_{j:06d}.txt',"w+") as f:
            f.write(f'multiples_code.neighbor_veto = {self.multiples_code.neighbor_veto}\n')
            f.write(f'multiples_code.neighbor_perturbation_limit = {self.multiples_code.neighbor_perturbation_limit}\n')
            f.write(f'multiples_code.retain_binary_apocenter = {self.multiples_code.retain_binary_apocenter}\n')
            f.write(f'multiples_code.wide_perturbation_limit = {self.multiples_code.wide_perturbation_limit}\n')
            f.write(f'initial conditions:\n')

    @property
    def stars(self):
        return self.multiples_code.stars

    @property
    def particles(self):
        return self.multiples_code.particles

    def init_smalln(self):
        self.SmallN = Smalln(convert_nbody=self.converter)

    def new_smalln(self):
        self.SmallN.reset()
        return self.SmallN

    def stop_smalln(self):
        self.SmallN.stop()

    def print_diagnostics(self, grav, energy_0=None):
        # Simple diagnostics.
        j = num
        energy_kinetic = grav.kinetic_energy
        energy_potential = grav.potential_energy
        (
            self.number_of_multiples,
            self.number_of_binaries,
            self.energy_in_multiples,
        ) = grav.get_total_multiple_energy()
        energy = energy_kinetic + energy_potential + self.energy_in_multiples
        print('')
        print(f'Time = {grav.get_time().in_(units.Myr)}')
        print(f'    top-level kinetic energy = {energy_kinetic}')
        print(f'    top-level potential energy = {energy_potential}')
        print(f'    total top-level energy = {energy_kinetic + energy_potential}')
        print(f'    {self.number_of_multiples} multiples, total energy = {self.energy_in_multiples}')
        print(f'    uncorrected total energy ={energy}')

        with open(f'log_test/log_{j:06d}.txt',"a+") as f:
            f.write(f'Time = {grav.get_time().in_(units.Myr)}\n')
            f.write(f'    top-level kinetic energy = {energy_kinetic}\n')
            f.write(f'    top-level potential energy = {energy_potential}\n')
            f.write(f'    total top-level energy = {energy_kinetic + energy_potential}\n')
            f.write(f'    {self.number_of_multiples} multiples, total energy = {self.energy_in_multiples}\n')
            f.write(f'    uncorrected total energy ={energy}\n')
    #        f.write(f'    {m} multiples formed so far \n')
        # Apply known corrections.

        energy_tidal = (
            grav.multiples_external_tidal_correction
            + grav.multiples_internal_tidal_correction
        )  # tidal error
        energy_error = grav.multiples_integration_energy_error  # integration error

        energy -= energy_tidal + energy_error
        print(f'    corrected total energy = {energy}')

        if energy_0 is not None: print('    relative energy error=', (energy-energy_0)/energy_0)

        return energy

    def evolve_model(self, t_end):
        j = num
    #    resolve, ignore = self.multiples_code.evolve_model(t_end)
    #    print(resolve, ignore)
        self.multiples_code.evolve_model(t_end)
        self.print_diagnostics(self.multiples_code, self.energy_0)
        self.model_time = self.multiples_code.model_time
    #    with open(f'log_test/log_{j:06d}.txt',"a+") as f:
    #        f.write(f'{resolve} encounters resolved\n')
     #       f.write(f'{ignore} encounters ignored\n')
    def stop(self):
        self.gravity.stop()
        self.kep.stop()
        self.stop_smalln()

#can vary parameters of cluster
number_of_stars = 5000
binary_frac = 1
number_of_high_mass_stars = 80
number_of_low_mass_stars = 4920
cut_off_high_mass = 0.2798
high_mass_stars = new_star_cluster(
   rand_seed = np.random.seed(num),
    stellar_mass = False,
    number_of_stars=number_of_high_mass_stars,
    # stellar_mass=False,
    initial_mass_function="kroupa",
    bin_initial_mass_function = "flat",
    lower_mass_limit = 8 | units.MSun,
    upper_mass_limit=100| units.MSun,
    effective_radius= .1 | units.parsec,
    star_distribution="plummer",
    star_distribution_w0=7.0, #density profile, how does this work?
    star_distribution_fd=2.0,
    star_metallicity=0.01,
    binary_frac = binary_frac,
    cut_off_high_mass = cut_off_high_mass, #| units.MSun,
#     mass_boundaries=[0.1,cut_off_high_mass, 100.0] | units.MSun,
#     alphas=[-0.3, -1.3, -4.6],
#     random=random,
)

low_mass_stars = new_star_cluster(
   rand_seed = np.random.seed(num),
    stellar_mass = False,
    number_of_stars=number_of_low_mass_stars,
    # stellar_mass=False,
    initial_mass_function="kroupa",
    bin_initial_mass_function = "flat",
    lower_mass_limit = 0.1 | units.MSun,
    upper_mass_limit=8| units.MSun,
    effective_radius= .3 | units.parsec,
    star_distribution="plummer",
    star_distribution_w0=7.0, #density profile, how does this work?
    star_distribution_fd=2.0,
    star_metallicity=0.01,
    binary_frac = binary_frac,
    cut_off_high_mass = cut_off_high_mass, #| units.MSun,
#     mass_boundaries=[0.1,cut_off_high_mass, 100.0] | units.MSun,
#     alphas=[-0.3, -1.3, -4.6],
#     random=random,
)


print("num stars", number_of_stars)
print("binary fraction", binary_frac)
print("top binary masses")
print("effective radius for high masses", 0.1)
print("effective radius for low masses", 0.3)
print("1 au binaries")
print("plummer distribution")

from amuse.support.console import set_printing_strategy
set_printing_strategy("custom", preferred_units=(units.pc, units.MSun, units.kms, units.Myr, units.J))

#numbers for various particle sets to be used later
num_binaries_and_stars_high = round((1 - (0.5*binary_frac))*number_of_high_mass_stars)
num_binaries_high = round((0.5*binary_frac)*number_of_high_mass_stars)
num_binary_stars_high = 2*num_binaries_high
num_single_stars_high = number_of_high_mass_stars - round( binary_frac*number_of_high_mass_stars)
num_single_stars_high

#numbers for various particle sets to be used later
num_binaries_and_stars_low = round((1 - (0.5*binary_frac))*number_of_low_mass_stars)
num_binaries_low = round((0.5*binary_frac)*number_of_low_mass_stars)
num_binary_stars_low = 2*num_binaries_low
num_single_stars_low = number_of_low_mass_stars - round( binary_frac*number_of_low_mass_stars)
num_single_stars_low

#new array was made to make colourmap for plots as one in cluster py doesn't work for individual stars
arr_high = np.array([1] * num_single_stars_high + [0] * num_binary_stars_high)
arr_low = np.array([1] * num_single_stars_low + [0] * num_binary_stars_low)

high_mass_cluster = high_mass_stars[0]
binary_components_high = high_mass_stars[1]
binaries_high = high_mass_stars[2]

low_mass_cluster = low_mass_stars[0]
binary_components_low = low_mass_stars[1]
binaries_low = low_mass_stars[2]


index_stars_binaries = 0
index_binary_component_stars = 1
index_binaries = 2


for child1, binary in zip(binary_components_high[0:num_binaries_high], binaries_high):
    child1.position = child1.position + binary.position
for child2, binary in zip(binary_components_high[num_binaries_high:num_binary_stars_high], binaries_high):
    child2.position = child2.position + binary.position

for child1, binary in zip(binary_components_high[0:num_binaries_high], binaries_high):
    child1.velocity = child1.velocity + binary.velocity
for child2, binary in zip(binary_components_high[num_binaries_high:num_binary_stars_high], binaries_high):
    child2.velocity = child2.velocity + binary.velocity

for child1, binary in zip(binary_components_low[0:num_binaries_low], binaries_low):
    child1.position = child1.position + binary.position
for child2, binary in zip(binary_components_low[num_binaries_low:num_binary_stars_low], binaries_low):
    child2.position = child2.position + binary.position

for child1, binary in zip(binary_components_low[0:num_binaries_low], binaries_low):
    child1.velocity = child1.velocity + binary.velocity
for child2, binary in zip(binary_components_low[num_binaries_low:num_binary_stars_low], binaries_low):
    child2.velocity = child2.velocity + binary.velocity


from amuse.units.constants import G

#default units
from amuse.support.console import set_printing_strategy
set_printing_strategy("custom", preferred_units=(units.pc, units.MSun, units.kms, units.Myr, units.J))

potential_energy = high_mass_stars[0].potential_energy(G=G)
virial_ratio = 0.5  # i.e. equilibrium, < 0.5 will collapse and > 0.5 will expand
scale_factor = np.sqrt(abs(virial_ratio*potential_energy) / high_mass_stars[0].kinetic_energy())
high_mass_stars[0].velocity *= scale_factor


pe = high_mass_stars[0].potential_energy()
#print(singles_and_binaries.kinetic_energy().value_in(units.J))
ke = high_mass_stars[0].kinetic_energy()
print("initial virial ratio for high mass stars",ke/pe)



potential_energy = low_mass_stars[0].potential_energy(G=G)
virial_ratio = 0.5  # i.e. equilibrium, < 0.5 will collapse and > 0.5 will expand
scale_factor = np.sqrt(abs(virial_ratio*potential_energy) / low_mass_stars[0].kinetic_energy())
low_mass_stars[0].velocity *= scale_factor

pe = low_mass_stars[0].potential_energy()
#print(singles_and_binaries.kinetic_energy().value_in(units.J))
ke = low_mass_stars[0].kinetic_energy()
print("initial virial ratio for low mass stars",ke/pe)

singles_and_binaries = Particles()
for star in low_mass_stars[0]:
  #  print(star)
    singles_and_binaries.add_particle(star)
for star in high_mass_stars[0]:
    singles_and_binaries.add_particle(star)


potential_energy = singles_and_binaries.potential_energy(G=G)
virial_ratio = 0.5  # i.e. equilibrium, < 0.5 will collapse and > 0.5 will expand
scale_factor = np.sqrt(abs(virial_ratio*potential_energy) / singles_and_binaries.kinetic_energy())
singles_and_binaries.velocity *= scale_factor

pe = singles_and_binaries.potential_energy()
#print(singles_and_binaries.kinetic_energy().value_in(units.J))
ke = singles_and_binaries.kinetic_energy()
print("initial virial ratio with binaries, virialised afterwards",ke/pe)


print("mass seg ratio", particle_attributes.mass_segregation_ratio(singles_and_binaries))

print("total mass of cluster", singles_and_binaries.mass.sum().value_in(units.MSun))
print("average velocity",sum(singles_and_binaries.velocity.lengths().value_in(units.kms))/len(singles_and_binaries))
#checking the lagrangian radius for the
total_mass = singles_and_binaries.mass.sum().value_in(units.MSun)
x, y, z = singles_and_binaries.center_of_mass()
print("virial radius", singles_and_binaries.virial_radius().value_in(units.pc))
lag = singles_and_binaries.LagrangianRadii(cm=(x,y,z),mf=(0.5,0.90,))
hmr = lag[0][0]
print("hmr", hmr)
lag_90 = lag[0][1]
print("lag_90", lag_90)

x, y, z = singles_and_binaries.center_of_mass()
half_mass = singles_and_binaries.mass.sum().value_in(units.MSun)/2
mass_total = singles_and_binaries.mass.sum()
x,y,z
com = np.array([x.value_in(units.pc),y.value_in(units.pc),z.value_in(units.pc)])
com
def vel_vec(star):
    """velocity vector of a given star"""
    return np.array([star.vx.value_in(units.kms), star.vy.value_in(units.kms), star.vx.value_in(units.kms)])
def pos_vec(star):
    """position vector of given star"""
    return  np.array([star.x.value_in(units.pc), star.y.value_in(units.pc), star.x.value_in(units.pc)])
#np.dot(pos_vec(cluster1[0][0]),vel_vec(cluster1[0][0]))


#adapted from https://stackoverflow.com/questions/39497496/how-do-i-retrieve-the-angle-between-two-vectors-3d
def angle(v1, v2):
    """angle between two vectors"""
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.rad2deg(angle)

radius = 0.3 | units.pc
number_of_stars = len(singles_and_binaries)
def get_rad(star):
    """gets the radius of the star from stars rather than snapshots"""
    rad= ((star.x)**2 + (star.y)**2
                 + (star.z**2))**0.5
    return rad
def get_rad_m(star):
    """gets the radius of the star from stars rather than snapshots"""
    rad= ((star.x.value_in(units.m))**2 + (star.y.value_in(units.m))**2
                 + (star.z.value_in(units.m))**2)**0.5
    return rad
def get_v(star):
    """gets the speed of the star from stars rather than snapshots"""
    v= ((star.vx.value_in(units.kms))**2 + (star.vy.value_in(units.kms))**2
                 + (star.vz.value_in(units.kms))**2)**0.5
    return v
def esc_vel(star):
    """escape velocity of the star for plummer model at radius r"""
    # G is in | m**3*kg**-1*s**-2
    return ((2 * constants.G * mass_total)/
             (get_rad(star)**2 +  hmr**2)**(1/2))**(1/2)

index = np.arange(number_of_stars)

stars = singles_and_binaries
stars.id = index + 1


G = 6.67428e-11 #| m**3*kg**-1*s**-2
def binding_energy( id1, id2):
    """binding energy of a given binary, takes the ids as input"""
    star1 =  model.multiples_code.stars[model.multiples_code.stars.id == id1 ]
    star2 = model.multiples_code.stars[model.multiples_code.stars.id == id2]
    a = get_orbital_elements_from_binaries(
    model.multiples_code.stars[model.multiples_code.stars.id == id1 ],
    model.multiples_code.stars[model.multiples_code.stars.id == id2])[2]
    #binding energy of binary
    E_tilda = -( constants.G * star1.mass *star2.mass ) / (2*a )
    #have to assume that the stars are equal masses here which i need to adapt when use a range of masses.
    # sigma is velocity dispersion, need to figure out how to get sigma
#    hard = abs(E_tilda) /(star1.mass * sigma**2)
    return E_tilda

def binary_binding_energy(binary):
    a = binary.semi_major_axis.value_in(units.m)
    return - (G * binary.child1_mass.value_in(units.kg) * binary.child2_mass.value_in(units.kg)) / 2*a


def mean_KE(stars):
    """returns the mean kinetic energy of the stars in the cluster"""
    return singles_and_binaries.kinetic_energy().value_in(units.J)/num_binaries_and_stars

def binary_hardness(binaries, stars):
    binary_hardness = []
    for binary in binaries:
        if abs(binary_binding_energy(binary)) > mean_KE(stars):
            binary_hardness.append("hard")
        else:
            binary_hardness.append("soft")
    return binary_hardness



def mean_KE2(stars):
    a = 0
    """returns the mean kinetic energy of the stars in the cluster"""
    single_KE_sum = singles_and_binaries[0:num_single_stars].kinetic_energy().value_in(units.J)
    for binary in singles_and_binaries[num_single_stars:num_binaries_and_stars]:
        a += 0.5* binary.mass.value_in(units.kg) * get_v(binary)**2
    total_ke = single_KE_sum + a
    return total_ke / num_binaries_and_stars




stars = singles_and_binaries



singles_and_binaries.remove_particles(binaries_high)
singles_and_binaries.remove_particles(binaries_low)

singles_and_binaries.add_particles(binary_components_high)
singles_and_binaries.add_particles(binary_components_low)

#potential_energy = singles_and_binaries.potential_energy(G=G)
#virial_ratio = 0.5  # i.e. equilibrium, < 0.5 will collapse and > 0.5 will expand
#scale_factor = np.sqrt(abs(virial_ratio*potential_energy) / singles_and_binaries.kinetic_energy())
#singles_and_binaries.velocity *= scale_factor

#radius = 1. | units.pc
number_of_stars = len(singles_and_binaries)
index = np.arange(number_of_stars)

stars = singles_and_binaries
stars.id = index + 1
stars.radius = (0.5 / len(singles_and_binaries)) | units.pc
print("test")

#print(singles_and_binaries.potential_energy().value_in(units.J))
pe = singles_and_binaries.potential_energy()
#print(singles_and_binaries.kinetic_energy().value_in(units.J))
ke = singles_and_binaries.kinetic_energy()
print("initial virial ratio",ke/pe)

lag = singles_and_binaries.LagrangianRadii(cm=(x,y,z),mf=(0.5,0.90,))
hmr_2 = 2* lag[0][0]
hmr_2.value_in(units.pc)
lag_90 = lag[0][1]
print("half mass radius",hmr_2)
hmr_2



print("imf")
for star in singles_and_binaries:
    print(star.mass.value_in(units.MSun), ",")


model = GravityWithMultiples(stars)
channel_from_code = model.multiples_code.stars.new_channel_to(stars)




converter = nbody_system.nbody_to_si(stars.mass.sum(), radius)
time_step = converter.to_si(0.01 | nbody_system.time)
#time_step = converter.to_si(5 | nbody_system.time)
print(f"One time step is {time_step.in_(units.Myr)}")


volume = 4/3 * 3.14 * lag[0][0].value_in(units.pc)**3
dens = (total_mass)/ volume
# print(dens)
# print(total_mass)
# print(lag[0][0])


hm_stars = []
colour_map = []
for val in singles_and_binaries.mass.value_in(units.MSun):
    if val < 10:
        colour_map.append(0)
    else:
        colour_map.append(1)
cmap_arr = np.array(colour_map)
cmap_arr


#this is for plotting the runaways for the animations
id_runaways =  [1, 48 ] #4614, 69, 2860, 17, 29, 296, 3808, 4445, 2922, 4211, 4222, 298, 2971, 3670, 3986, 3383, 4446, 2625, 57, 2584, 3019, 3557, 4943, 2668, 3026, 3062, 3519, 4392, 4417, 2758, 3195, 4007, 4902, 3834, 4709, 550, 2744, 3319, 3336, 4102, 1806, 3325, 3362, 4100, 4519, 3162, 3983, 4128, 1097, 2945, 3193, 3306, 4620, 4635, 2756, 3235, 3566, 559, 2823, 3883, 4449, 2910, 4266, 1348, 1814, 4985, 876, 1059, 363, 2833, 3395, 4455, 485, 1523, 735, 4274, 4487, 3874, 4306, 533, 3934, 4112, 2771, 4706, 3341, 3589, 2773, 2632, 1872, 4782, 2249, 4488, 3010, 3940, 2246, 2322, 3898, 4403, 13, 53, 2027, 3498, 4386, 4039, 4092, 4792, 10, 50, 2194, 4654, 3681]
id_runaways_minus_1 = list(ele -1 for ele in id_runaways )
arr = np.concatenate((arr_high, arr_low), axis = 0)


num_binaries_and_stars = round((1 - (0.5*binary_frac))*number_of_stars)
num_binaries = round((0.5*binary_frac)*number_of_stars)
num_binary_stars = 2*num_binaries
num_single_stars = number_of_stars - round( binary_frac*number_of_stars)
num_single_stars

#ids of the primaries and secondaries
child1_ids = stars[num_single_stars:round(num_single_stars + num_binaries)].id
child2_ids = stars[round(num_single_stars + num_binaries):number_of_stars].id



#save the initial model
singles_and_binaries.savepoint(model.model_time)

time_steps = []
multiples_info = []
no_runaways = 0
id_runaways = []
time_runaways = []
ims = []
hm_density = []
j = 0


#while no_runaways < 20:
while model.model_time <0.5 | units.myr:
    j=j+1
    x,y,z = singles_and_binaries.center_of_mass()
    lag = singles_and_binaries.LagrangianRadii(cm=(x,y,z),mf=(0.5,0.9,))
    lag_90 = lag[0][1]
    for star in singles_and_binaries:
        #runaways have to be twice the HMR (or something else) away and have vel higher than esc vel and
        #also need velocity direction to be pointing away from the cluster, dot the v vector and r vector
        # and see if the angle is smaller than some given amount.5
        if get_rad(star) > lag_90:
            if get_v(star) > esc_vel(star).value_in(units.kms):
                if star.id not in id_runaways:
                    if angle(pos_vec(star),vel_vec(star)) < 10:
                        #is the star part of a binary? if so should have id larger than the number of single stars
                        #if it's not, then i can just use the old criteria.
                        if star.id < num_single_stars:
                            no_runaways += 1
                            print("runaway!")

                            print(star.id)
                            id_runaways.append(star.id)
                            time_runaways.append(model.model_time)
                        if star.id >= num_single_stars:
                            #then it's probs part of a binary system, but will check if it's close enough
                            #to it's binary pair.
                            #first, need to indentify the other child, find the closest one
                            #child = particle_attributes.find_closest_particle_to(singles_and_binaries,
                            #                                                     star.x,star.y,star.z)
                            child = (particle_attributes.nearest_neighbour(stars, neighbours=None,
                                                                           max_array_length=10000000)[star.id - 1])
                            #now work out the distance between star.id and the child star.
                            distance = ((star.x.value_in(units.au) - child.x.value_in(units.au) )**2 +
                                        (star.y.value_in(units.au)  - child.y.value_in(units.au) )**2
                                  +(star.z.value_in(units.au)- child.z.value_in(units.au) )**2)**(0.5)
                            #if the distance is less than 100 au then it still counts as a binary
                            if distance > 1.1:
                                #then it's probably not in a binary and counts as a runaway
                                no_runaways += 1
                                print("runaway!")

                                print(star.id)
                                id_runaways.append(star.id)
                                time_runaways.append(model.model_time)
                            if distance <= 1.1:
                                #then it still counts as a binary. need to see if the binary com is moving away.

                                new_binary = Particles()
                                new_binary.add_particle(star)
                                new_binary.add_particle(child)
                                comv = particle_attributes.center_of_mass_velocity(new_binary)
                                coms = (comv[0].value_in(units.kms)**2 + comv[1].value_in(units.kms)**2 +
                                        comv[2].value_in(units.kms)**2)**0.5
                                binary_pair = [star.id, child.id]
                                if coms > esc_vel(star).value_in(units.kms):
                                    if binary_pair[0] not in id_runaways:
                                        if binary_pair[1] not in id_runaways:
                                            if angle(pos_vec(star),comv.value_in(units.kms)) < 10:

                                                no_runaways += 2
                                                print("runaway binary!")

                                                print(star.id,child.id)
                                                id_runaways.append(star.id)
                                                id_runaways.append(child.id)
                                                time_runaways.append(model.model_time)


    #to calculate density of hmr
    volume = 4/3 * 3.14 * lag[0][0].value_in(units.pc)**3
    dens = (total_mass/2)/ volume
    hm_density.append(dens)
    time_steps.append(model.model_time)
    model.evolve_model(model.model_time + time_step)
    channel_from_code = model.multiples_code.stars.new_channel_to(singles_and_binaries)
    channel_from_code.copy()
    singles_and_binaries.savepoint(model.multiples_code.model_time)
    m = model.multiples_code.print_multiples()
    multiples_info.append([model.number_of_multiples,
            model.number_of_binaries,
            model.energy_in_multiples, m, model.model_time])
  # simple plotting routine

    fig, ax = plt.subplots()

   # plt.title("5000 stars in VE until 5 runaways form, 0.5 binary frac, x-y, t = %g Myr" % model.model_time.value_in(units.myr))
    plt.title("5000 stars evolved until 0.5 Myr, new mass seg (0.1,0.3), \n high masses in 1 AU binaries 1 binary fraction = %g Myr" % model.model_time.value_in(units.myr))

    ax.scatter(singles_and_binaries.x.value_in(units.pc),
               singles_and_binaries.y.value_in(units.pc), s=2,
             #  c = cluster1[0].mass.value_in(units.MSun))
               c = arr, cmap = "bwr")
#    ax.scatter(singles_and_binaries[id_runaways_minus_1].x.value_in(units.pc),
#            singles_and_binaries[id_runaways_minus_1].y.value_in(units.pc),
#            s=12,
#            edgecolors="none", c ='red',
#            )
             # c = 'b')

#     ax.scatter(binary_stars_x, binary_stars_y, s=2,
#              #  c = cluster1[0].mass.value_in(units.MSun))
#                c = "green")

#     plt.title("test population with binaries in VE until 5 runaways form, x-y, t = %g Myr" % model.model_time.value_in(units.myr))
#     ax.scatter(run_x_arr, run_y_arr, s=2,
#              #  c = cluster1[0].mass.value_in(units.MSun))
#               # c = cmap_arr,
#                cmap = "cool")
#     ax.scatter(hmx_arr, hmy_arr, s=2,
#              #  c = cluster1[0].mass.value_in(units.MSun))
#               # c = cmap_arr,
#                cmap = "cool")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel('x/pc')
    ax.set_ylabel('y/pc')
#    ax.scatter(
#           cluster1[0][binaries_minus_1].x.value_in(units.pc),
#           cluster1[0][binaries_minus_1].y.value_in(units.pc),
#           edgecolors="none", c ='green',
#            )
    plt.legend(['stars','runaways', 'binary'], loc = 'upper right')
    im = plt.savefig(fname = f'1au_mass_seg_0.1_3/figs_{j:06d}.png')
    ims.append([im])



#get all the data from the snapshots in the model history
xdata = []
ydata = []
zdata = []

vxdata = []
vydata = []
vzdata = []
times = []

KE = []
PE = []

no_snaps = 0
for snapshot in singles_and_binaries.history:
    xdata.append(snapshot.x)
    ydata.append(snapshot.y)
    zdata.append(snapshot.z)

    vxdata.append(snapshot.vx)
    vydata.append(snapshot.vy)
    vzdata.append(snapshot.vz)
    KE.append(snapshot.kinetic_energy().value_in(units.J))
    PE.append(snapshot.potential_energy().value_in(units.J))

    no_snaps = no_snaps+1

print("random seed", num)
print("number of runaways", len(id_runaways))

print("runaway ids",id_runaways)
id_runaways_minus_1 = list(ele -1 for ele in id_runaways )
#print(id_runaways_minus_1)
print("runaway masses in MSun")
#print(snapshot[snapshot.id == id_runaways[0]].mass.value_in(units.MSun),
#      snapshot[snapshot.id == id_runaways[1]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[2]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[3]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[4]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[5]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[6]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[7]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[8]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[9]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[10]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[11]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[12]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[13]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[14]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[15]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[16]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[17]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[18]].mass.value_in(units.MSun),
#snapshot[snapshot.id == id_runaways[19]].mass.value_in(units.MSun))


print("masses below")
for i in id_runaways:
    print(snapshot[snapshot.id == i].mass.value_in(units.MSun)[0], ',')
    #print(get_v(snapshot[snapshot.id == i])[0], ',')
print("velocities below")
for i in id_runaways:
    #print(snapshot[snapshot.id == i].mass.value_in(units.MSun)[0], ',')
    print(get_v(snapshot[snapshot.id == i])[0], ',')
#print("runaway velocities in km/s")
#print(get_v(snapshot[snapshot.id == id_runaways[0]]),get_v(snapshot[snapshot.id == id_runaways[1]]),
#      get_v(snapshot[snapshot.id == id_runaways[2]]),get_v(snapshot[snapshot.id == id_runaways[3]]),
#      get_v(snapshot[snapshot.id == id_runaways[4]]),get_v(snapshot[snapshot.id == id_runaways[5]]),
#      get_v(snapshot[snapshot.id == id_runaways[6]]),get_v(snapshot[snapshot.id == id_runaways[7]]),
#      get_v(snapshot[snapshot.id == id_runaways[8]]),get_v(snapshot[snapshot.id == id_runaways[9]]),
#      get_v(snapshot[snapshot.id == id_runaways[10]]),get_v(snapshot[snapshot.id == id_runaways[11]]),
#      get_v(snapshot[snapshot.id == id_runaways[12]]),get_v(snapshot[snapshot.id == id_runaways[13]]),
#      get_v(snapshot[snapshot.id == id_runaways[14]]),get_v(snapshot[snapshot.id == id_runaways[15]]),
#      get_v(snapshot[snapshot.id == id_runaways[16]]),get_v(snapshot[snapshot.id == id_runaways[17]]),
#      get_v(snapshot[snapshot.id == id_runaways[18]]),get_v(snapshot[snapshot.id == id_runaways[19]]))
print("runaway times")
for i in time_runaways:
    print(i.value_in(units.Myr), ',')


time = []
for i in time_steps:
    time.append(i.value_in(units.myr))

fig = plt.figure()

plt.plot(time, hm_density)

plt.title("Density of the core")
plt.xlabel("Time / Myr")
plt.ylabel("Density / MSun/pc**3")
plt.legend(['with sts 0.5'], loc = 'upper right')
plt.show()
im = plt.savefig(fname = f'1au_mass_seg_0.1_3/density.png')

