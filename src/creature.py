import genome 
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import pybullet as p
import numpy as np

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq, control_phase):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        # Added phase control capability
        self.phase = control_phase
    

    def get_output(self):
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = 1
            else:
                output = -1
            
        if self.motor_type == MotorType.SINE:
            output = np.sin(self.phase)
        
        return output 

class Creature:
    def __init__(self, gene_count):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        
        # Initialize values upon creature creation
        self.dist_score = 0
        self.contact_score = 0
        self.height_score = 0
        self.fitness = 0
        self.body_id = None
        self.mountain_id = None


    def get_flat_links(self):
        if self.flat_links == None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links
    
    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        
        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(self.flat_links[0], 
                                self.flat_links[0].name, 
                                self.flat_links, 
                                exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "pepe") #  choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors == None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                m = Motor(l.control_waveform, l.control_amp,  l.control_freq, l.control_phase)
                motors.append(m)
            self.motors = motors 
        return self.motors 
    
    def update_position(self, pos):
        if self.start_position == None:
            self.start_position = (7, 0, 0)
        else:
            self.last_position = pos

    def get_fitness(self):
        if self.body_id is None or self.body_id < 0 or self.start_position is None:
            return 0.0

        creature_pos, creature_orient = p.getBasePositionAndOrientation(self.body_id)
        mountain_pos, _ = p.getBasePositionAndOrientation(self.mountain_id)
        self.update_position(creature_pos)

        self.get_expanded_links()
        num_links = len(self.exp_links)
        total_mass = sum(np.pi * (l.link_radius ** 2) * l.link_length for l in self.exp_links)

        # Basic movement metrics
        contact_points = p.getContactPoints(self.body_id, self.mountain_id)
        has_contact = len(contact_points) > 0
        xy_disp = np.linalg.norm(np.array(creature_pos[:2]) - np.array(self.start_position[:2]))
        height_gain = creature_pos[2] - self.start_position[2]

        if num_links < 2 or xy_disp < 0.5:
            return 0.1  

        # Normalization constants
        max_height = 5    
        max_dist = 10 
        
        # 1. Contact with mountain at height (20 points max)
        base_dist_to_mountain = np.linalg.norm(np.array(mountain_pos[:2]) - np.array(creature_pos[:2]))
        if has_contact:
            # Exponential scaling with height 
            contact_multiplier = (1 + creature_pos[2]) ** 1.5
            contact_score = np.clip(contact_multiplier * 8, 0, 20)  # Max 40 points
        else:
            contact_score = -20  # Penalty for not touching mountain
        
        # 2. Reward getting close to mountain (20 points max)
        if creature_pos[2] < 1.0: 
            proximity_score = np.clip(1 - base_dist_to_mountain / max_dist, 0, 1) * 20
        else:  
            proximity_score = np.clip(1 - base_dist_to_mountain / max_dist, 0, 1) * 5
        
        # 3. Reward staying upright (5 points max)
        quat = creature_orient
        uprightness = 1 - abs(quat[0]) - abs(quat[1]) 
        upright_score = np.clip(uprightness, 0, 1) * 5
        
        # 4. Reward moving toward mountain (10 points max)
        directional_score = 0
        if creature_pos[2] < 1.5:
            mountain_direction = np.array(mountain_pos[:2]) - np.array(self.start_position[:2])
            creature_movement = np.array(creature_pos[:2]) - np.array(self.start_position[:2])
            
            if np.linalg.norm(mountain_direction) > 0.1 and np.linalg.norm(creature_movement) > 0.1:
                # Dot product gives directional alignment
                alignment = np.dot(mountain_direction, creature_movement) / (
                    np.linalg.norm(mountain_direction) * np.linalg.norm(creature_movement)
                )
                directional_score = np.clip(alignment, -1, 1) * 10
        
        # # 5. Small reward for any movement (5 points max)
        exploration_score = np.clip(xy_disp / 3, 0, 1) * 5
        
        # 6. Bonus for climbing progress - height gained while in contact (40 points max)
        climbing_progress_score = 0
        if has_contact and height_gain > 0:
            climbing_progress_score = np.clip(height_gain / max_height, 0, 1) * 40
        
        # Combine all positive scores (max 100 points from positive rewards)
        fitness = (contact_score + proximity_score + exploration_score + upright_score +
                directional_score  + climbing_progress_score)
        
        # PENALTIES

        # Mass penalty - discourage overly heavy creatures
        if total_mass > 8:
            fitness -= (total_mass - 8) * 2
        
        # Complexity penalty - discourage overly complex creatures
        if num_links > 11:
            fitness -= (num_links - 11) * 3.3
        
        # Tiny creature penalty - discourage creatures that are too small to be functional
        if total_mass < 0.05:
            fitness -= 10
        
        # Distance penalty - heavily penalize creatures that go very far from mountain
        if base_dist_to_mountain > 12:
            fitness -= (base_dist_to_mountain - 12) * 4
        
        # Height without contact penalty - discourage cheating by getting high without climbing
        if height_gain > 1.0 and not has_contact:
            fitness -= height_gain * 8
        
        # Ensure fitness is not negative and capped at 100
        fitness = np.clip(fitness, 0, 100)
        
        self.fitness = fitness
        return fitness


    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None