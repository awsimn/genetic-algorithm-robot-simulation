import pybullet as p
import time
import creature
import genome
import numpy as np
import simulation

# Initialize the simulation in GUI mode
sim = simulation.Simulation(0, p.GUI)

def get_total_mass(body_id):
    total_mass = 0
    num_joints = p.getNumJoints(body_id)

    # Base (link -1)
    dynamics = p.getDynamicsInfo(body_id, -1)
    total_mass += dynamics[0]  # [0] is mass

    # All other links
    for i in range(num_joints):
        dynamics = p.getDynamicsInfo(body_id, i)
        total_mass += dynamics[0]

    return total_mass

sim.create_floor()
# Create the environment (loads the mountain) and get its ID
mountain = sim.create_environment()

# Load evolved creature from file
cr = creature.Creature(gene_count=3)
cr.dna = genome.Genome.from_csv('elite_cr3.csv')

# Write creature to URDF and load it into the simulation
with open('test.urdf', 'w') as f:
    f.write(cr.to_xml())

robot_id = p.loadURDF('test.urdf', (7, 0, 1))

m=get_total_mass(robot_id)  # Check total mass of the robot
print(f"Total mass of the robot: {m} kg")

# Set up gravity and real-time simulation
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)  # we'll manually step

time.sleep(2)  # Allow some time for the simulation to initialize

# Simulation loop
step = 0
while True:
    p.stepSimulation()

    
        # Update motors every 24 steps (~0.1 seconds)
    if step % 24 == 0:
        sim.update_motors(robot_id, cr)

    time.sleep(1.0 / 240.0)
    step += 1
