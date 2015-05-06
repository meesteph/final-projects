#################################################################
#						PHY 480 Final Project					#
#		Title: N-Body Gravitationally Interacting System 		#
#		Date:  May 5th, 2015									#
#		Description: Using the Runge-Kutta 4th order			#
#		integrator, a gravitationally interacting system of 	#
#		particles is initialized with varying distributions.	#
#################################################################

from __future__ import division
import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.backends.backend_pdf import PdfPages

class Particle(object):

	def __init__(self, mass, x, y, z, vx, vy, vz):
		self.mass = mass
		position = np.array([x,y,z])
		velocity = np.array([vx,vy,vz])

		self.position = position
		self.velocity = velocity

		self.position_list = [position]
		self.velocity_list = [velocity]

	def __str__(self):
		return "Particle of mass {}, located at {}, velocity of {}".format(self.mass, self.position, self.velocity)

	def __repr__(self):
		return self.__str__()

	def __add__(self, particle2):
		new_mass = self.m + particle2.m;
		new_vx = (self.m * self.velocity[0] + particle2.m * particle2[0]) / new_mass
		new_vy = (self.m * self.velocity[1] + particle2.m * particle2[1]) / new_mass
		new_vz = (self.m * self.velocity[2] + particle2.m * particle2[2]) / new_mass
		new_x = (self.x + particle2.x) / 2.0
		new_y = (self.y + particle2.y) / 2.0
		new_z = (self.z + particle2.z) / 2.0

		return Particle(new_mass, new_x, new_y, new_z, new_vx, new_vy, new_vz)

	def get_mass(self):
		return self.mass

	def get_position_vector(self):
		return self.position

	def get_velocity_vector(self):
		return self.velocity;

	def get_velocity_magnitude(self):
		return (self.velocity[0] ** 2.0 + self.velocity[1] ** 2.0 + self.velocity[2] ** 2.0) ** (1.0 / 2.0)

	def get_distance(self, particle2):
		x_diff = self.position[0] - particle2.position[0]
		y_diff = self.position[1] - particle2.position[1]
		z_diff = self.position[2] - particle2.position[2]
		return np.array([x_diff, y_diff, z_diff])

	def update_velocity(velocity_vector):

		self.velocity = velocity_vector

	def update_position(position_vector):

		self.position = position_vector


	def combine_particles(self, particle2, particle_list, crit_radius):
		"""
		Called as a check to see if particles have collided, if so then the two combine into a new particle.
		"""

		distance = self.get_distance(particle2)
		r = (distance[0] ** 2.0 + distance[1] ** 2.0 + distance[2] ** 2.0) ** (1.0 / 2.0)

		if r < crit_radius:

			print("Combining Particles")

			M = self.get_mass() + particle2.get_mass()
			new_v = (self.get_mass() * self.get_velocity_vector() + particle2.get_mass() * particle2.get_velocity_vector()) / M
			new_p = (self.get_position_vector() + particle2.get_position_vector()) / 2.0

			new_particle = Particle(M, new_p[0], new_p[1], new_p[2], new_v[0], new_v[1], new_v[2])
			particle_list.append(new_particle)

			particle_list.remove(self)
			particle_list.remove(particle2)


def net_acceleration(particle1, particle_list):
	"""
	Calculates the net acceleration on particle1 due to every other particle in the list.
	Arguments:
		particle1 		:=	particle being accelerated
		particle_list	:= 	list of all particles
	"""

	G = 1
	a_sum = np.array([0,0,0])

	for i in range(len(particle_list)):

		if particle_list[i] != particle1:

			mass = particle_list[i].get_mass()
			r = particle1.get_distance(particle_list[i])

			r_mag = (r[0] ** 2.0 + r[1] ** 2.0 + r[2] ** 2.0) ** (1.0 / 2.0)
			r_hat = r / r_mag

			a = -G * mass * r_hat / (r_mag ** 2.0)
			a_sum += a

	return a_sum

def initial_velocity(m1, m2, r, a):
	"""
	Calculates initial velocity for a given particle.
	Arguments:
		m1 	:=	 mass of star
		m2	:=	 mass of planet
		r 	:=	 distance between m1 and m2
		a 	:=	 semimajor axis
	"""

	G = 1
	p1 = G * (m1 + m2)
	p2 = 2.0 / r - 1.0 / a

	return (p1 * p2) ** (1.0 / 2.0)


def initialize_particles(distribution, N):
	"""
	Initializes the particle list for a given distribution.
	Arguments:
		distribution 	:=	string indicating desired distribution of particles
		N 				:=	number of particles in distribution
	"""

	G = 1

	particle_list = []

	if distribution == "moon":

		star_mass = 1000.0
		star = Particle(star_mass, 0, 0, 0, 0, 0, 0)
		particle_list.append(star)

		planet_mass = 5.0
		planet_v = (G * star_mass / 6) ** (1.0 / 2.0)
		planet = Particle(planet_mass, 6, 0, 0, 0, planet_v, 0)
		particle_list.append(planet)

		moon_mass = 0.1
		moon_v = (G * planet_mass / 0.35) ** (1.0 / 2.0)
		moon = Particle(moon_mass, 6.35, 0, 0, 0, planet_v, moon_v)
		particle_list.append(moon)



	if distribution == "random":

		for i in range(N):

			M = random.uniform(100.0,1000.0)

			x = random.uniform(0.0,10.0)
			y = random.uniform(0.0,10.0)
			z = random.uniform(0.0,10.0)

			vx = random.uniform(-5.0, 5.0)
			vy = random.uniform(-5.0, 5.0)
			vz = random.uniform(-5.0, 5.0)

			particle = Particle(M, x, y, z, vx, vy, vz)
			particle_list.append(particle)

	if distribution == "eccentric":

		star_mass = 1000.0
		particle = Particle(star_mass, 0, 0, 0, 0, 0, 0)
		particle_list.append(particle)

		for i in range(N):

			mass = random.uniform(0.0,1.0)

			x = random.uniform(3.0, 10.0)
			y = 0
			z = 0

			r = (x ** 2.0 + y ** 2.0 + z ** 2.0) ** (1.0 / 2.0)
			a = r - 1.0

			vx = 0
			vy = initial_velocity(star_mass, mass, r, a)
			vz = 0

			particle = Particle(mass, x, y, z, vx, vy, vz)
			particle_list.append(particle)

	if distribution == "solar":

		star_mass = 100.0
		particle = Particle(star_mass, 0, 0, 0, 0, 0, 0)
		particle_list.append(particle)

		for i in range(N):

			mass = random.uniform(0.0,0.1)

			x = random.uniform(2.0,10.0)
			y = 0
			z = 0

			r = (x ** 2.0 + y ** 2.0 + z ** 2.0) ** (1.0 / 2.0)

			vx = 0
			vy = (G * (mass + star_mass) / r) ** (1.0 / 2.0)
			vz = 0

			particle = Particle(mass, x, y, z, vx, vy, vz)

			particle_list.append(particle)

	return particle_list

def KE(particle):
	"""
	Returns kinetic energy of the particle passed in.
	"""

	m = particle.get_mass()
	v = particle.get_velocity_magnitude()

	return (1.0 / 2.0) * m * v ** 2.0

def PE(particle1, particle2):
	"""
	Returns gravitational potential energy between both particles passed in.
	"""

	G = 1

	m1 = particle1.get_mass()
	m2 = particle2.get_mass()

	r = particle1.get_distance(particle2)
	r = (r[0] ** 2.0 + r[1] ** 2.0 + r[2] ** 2.0) ** (1.0 / 2.0)

	return -G * m1 * m2 / r

def totalE(particle_list):
	"""
	Returns the sum of kinetic and potential energies for all particles in the list.
	"""

	KE_sum = 0
	PE_sum = 0

	for i in range(len(particle_list)):

		K = KE(particle_list[i])

		for j in range(i, len(particle_list)):

			if i != j:

				P = PE(particle_list[i], particle_list[j])
				PE_sum += P

		KE_sum += K

	return PE_sum + KE_sum, PE_sum, KE_sum

def rk4(particle1, particle_list, h):
	"""
	Integrator function. Returns a new velocity and new position according to the laws of motion.
	Arguments:
		particle1		:=	particle for which new position and velocity are being calculated
		particle_list	:=	list of all particles
		h 				:=	step size
	"""

	dummy = particle1

	v_i = particle1.get_velocity_vector()
	x_i = particle1.get_position_vector()

	k_r1 = v_i
	k_v1 = net_acceleration(dummy, particle_list)

	k_r2 = v_i + k_v1 * h / 2.0
	dummy.position = dummy.get_position_vector() + k_r1 * h / 2.0
	k_v2 = net_acceleration(dummy, particle_list)

	k_r3 = v_i + k_v2 * h / 2.0
	dummy.position = dummy.get_position_vector() + k_r2 * h / 2.0
	k_v3 = net_acceleration(dummy, particle_list)

	k_r4 = v_i + k_v3 * h
	dummy.position = dummy.get_position_vector() + k_r3 * h
	k_v4 = net_acceleration(dummy, particle_list)

	new_velocity = v_i + (h / 6.0) * (k_v1 + 2.0 * k_v2 + 2.0 * k_v3 + k_v4)
	new_position = x_i + (h / 6.0) * (k_r1 + 2.0 * k_r2 + 2.0 * k_r3 + k_r4)

	return new_velocity, new_position

def integrate(particle_list):
	"""
	Calls rk4 on every particle in the system for multiple timesteps. Returns lists of kinetic, potential, and
	total energy of the system for a given timestep.
	"""

	energy_list = []
	time_list = []
	KE_list = []
	PE_list = []

	start_t = 0.001
	h = 0.00001
	end_t = 3.41

	crit_radius = 0.1

	while start_t < end_t:

		vp_list = []

		for i in range(len(particle_list)):

			particle1 = particle_list[i]

			new_velocity, new_position = rk4(particle1, particle_list, h)

			vp_list.append([new_velocity, new_position])

		for i in range(len(particle_list)):

			particle_list[i].velocity = vp_list[i][0]
			particle_list[i].position = vp_list[i][1]

			particle_list[i].velocity_list.append(vp_list[i][0])
			particle_list[i].position_list.append(vp_list[i][1])

		for i in range(len(particle_list)):

			for j in range(i+1, len(particle_list)):

				try:

					particle_list[i].combine_particles(particle_list[j], particle_list, crit_radius)

				except IndexError:

					break

		energy, P, K = totalE(particle_list)
		energy_list.append(energy)
		PE_list.append(P)
		KE_list.append(K)



		start_t += h
		time_list.append(start_t)

		print start_t

	return time_list, energy_list, PE_list, KE_list


def main():

	particle_list = initialize_particles("moon", 1)

	for i in range(len(particle_list)):

		print particle_list[i]

	time_list, energy_list, PE_list, KE_list = integrate(particle_list)

	for i in range(len(particle_list)):

		print particle_list[i]

	pp = PdfPages("SolarOrbitTrajectoryTest.pdf")
	plt.title("Lunar Orbit Trajectory Test")
	plt.xlabel("X")
	plt.ylabel("Y")

	for i in range(len(particle_list)):

		current_particle = particle_list[i]

		positions = current_particle.position_list

		Xs = []
		Ys = []

		for j in range(len(positions)):

			Xs.append(positions[j][0])
			Ys.append(positions[j][1])

		if i == 1:
			plt.plot(Xs, Ys, "b")
		else:
			plt.plot(Xs, Ys, "g")

	plt.plot(0, 0, "o")

	pp.savefig()
	plt.show()
	pp.close()

	pp = PdfPages("Energy.pdf")
	plt.title("Energy vs Timestep")
	plt.xlabel("Timestep")
	plt.ylabel("Energy")
	plt.plot(time_list, energy_list, "r")
	pp.savefig()
	plt.close()
	pp.close()

	pp2 = PdfPages("BothEnergies.pdf")
	plt.title("KE and PE for lunar orbit")
	plt.xlabel("Timestep")
	plt.ylabel("Energy")
	plt.plot(time_list, KE_list, "r")
	plt.plot(time_list, PE_list, "b")
	plt.plot(time_list, energy_list, "k")
	pp2.savefig()
	plt.close()
	pp2.close()


main()

