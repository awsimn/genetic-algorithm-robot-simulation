import numpy as np
import copy 
import random

class Genome():
    @staticmethod 
    def get_random_gene(length):
        gene = np.array([np.random.random() for i in range(length)])
        return gene
    
    @staticmethod 
    def get_random_genome(gene_length, gene_count):
        genome = [Genome.get_random_gene(gene_length) for i in range(gene_count)]
        return genome

    @staticmethod
    def get_gene_spec():
        gene_spec =  {"link-shape":{"scale":1}, 
            "link-length": {"scale":0.75},
            "link-radius": {"scale":0.6},
            "link-recurrence": {"scale":2},
            "link-mass": {"scale":0.8},
            "joint-type": {"scale":1},
            "joint-parent":{"scale":1},
            "joint-axis-xyz": {"scale":1},
            "joint-origin-rpy-1":{"scale":np.pi},
            "joint-origin-rpy-2":{"scale":np.pi},
            "joint-origin-rpy-3":{"scale":np.pi},
            "joint-origin-xyz-1":{"scale":0.5},
            "joint-origin-xyz-2":{"scale":0.5},
            "joint-origin-xyz-3":{"scale":0.4},
            "control-waveform":{"scale":1},
            "control-amp":{"scale":0.4},
            "control-freq":{"scale":1.2},
            "control-phase":{"scale":np.pi * 2},
            "grip-strength":{"scale":20},
            "body-part-type":{"scale":1},
            "attachment-point":{"scale":1},
            "surface-friction":{"scale":2.0},
            "joint-friction":{"scale":1.0}
            }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec
    
    @staticmethod
    def get_gene_dict(gene, spec):
        gdict = {}
        for key in spec:
            ind = spec[key]["ind"]
            scale = spec[key]["scale"]
            gdict[key] = gene[ind] * scale
        return gdict

    @staticmethod
    def get_genome_dicts(genome, spec):
        gdicts = []
        for gene in genome:
            gdicts.append(Genome.get_gene_dict(gene, spec))
        return gdicts

    @staticmethod
    def expandLinks(parent_link, uniq_parent_name, flat_links, exp_links):
        children = [l for l in flat_links if l.parent_name == parent_link.name]
        sibling_ind = 1
        for c in children:
            for r in range(int(c.recur)):
                sibling_ind  = sibling_ind +1
                c_copy = copy.copy(c)
                c_copy.parent_name = uniq_parent_name
                uniq_name = c_copy.name + str(len(exp_links))
                #print("exp: ", c.name, " -> ", uniq_name)
                c_copy.name = uniq_name
                c_copy.sibling_ind = sibling_ind
                exp_links.append(c_copy)
                assert c.parent_name != c.name, "Genome::expandLinks: link joined to itself: " + c.name + " joins " + c.parent_name 
                Genome.expandLinks(c, uniq_name, flat_links, exp_links)

    @staticmethod
    def genome_to_links(gdicts):
        links = []
        link_ind = 0
        parent_names = [str(link_ind)]
        parent_links = []
        
        for gdict in gdicts:
            link_name = str(link_ind)
            
            if link_ind == 0:
                # First link is always the main body
                parent_name = "None"
                gdict["link-length"] = max(gdict["link-length"], 0.8)  # Ensure main body is substantial
                gdict["link-radius"] = max(gdict["link-radius"], 0.5)
            else:
                # Determine parent based on body part type
                body_part_type = gdict["body-part-type"]
                if body_part_type < 0.2 and len(parent_names) > 1:  # Torso extension
                    parent_name = parent_names[0]  # Always attach to main body
                elif body_part_type < 0.8:  # Limb
                    # Attach to main body or other torso parts
                    available_parents = [p for i, p in enumerate(parent_names) if i < min(3, len(parent_names))]
                    parent_ind = int(gdict["joint-parent"] * len(available_parents))
                    parent_name = available_parents[parent_ind]
                else:  # Extremity
                    # Attach to limbs or torso parts, avoid main body if possible
                    if len(parent_names) > 2:
                        available_parents = parent_names[1:]  # Skip main body
                        parent_ind = int(gdict["joint-parent"] * len(available_parents))
                        parent_name = available_parents[parent_ind]
                    else:
                        parent_ind = int(gdict["joint-parent"] * len(parent_names))
                        parent_name = parent_names[parent_ind]
            
            recur = gdict["link-recurrence"]
            link = URDFLink(name=link_name, 
                            parent_name=parent_name, 
                            recur=recur+1, 
                            link_length=gdict["link-length"], 
                            link_shape=gdict["link-shape"],
                            link_radius=gdict["link-radius"], 
                            link_mass=gdict["link-mass"],
                            joint_type=gdict["joint-type"],
                            joint_parent=gdict["joint-parent"],
                            joint_axis_xyz=gdict["joint-axis-xyz"],
                            joint_origin_rpy_1=gdict["joint-origin-rpy-1"],
                            joint_origin_rpy_2=gdict["joint-origin-rpy-2"],
                            joint_origin_rpy_3=gdict["joint-origin-rpy-3"],
                            joint_origin_xyz_1=gdict["joint-origin-xyz-1"],
                            joint_origin_xyz_2=gdict["joint-origin-xyz-2"],
                            joint_origin_xyz_3=gdict["joint-origin-xyz-3"],
                            control_waveform=gdict["control-waveform"],
                            control_amp=gdict["control-amp"],
                            control_freq=gdict["control-freq"],
                            control_phase=gdict["control-phase"],
                            grip_strength=gdict["grip-strength"],
                            body_part_type=gdict["body-part-type"],
                            attachment_point=gdict["attachment-point"],
                            surface_friction=gdict["surface-friction"],
                            joint_friction=gdict["joint-friction"])
            links.append(link)
            parent_links.append(link)
            if link_ind != 0:# don't re-add the first link
                parent_names.append(link_name)
            link_ind = link_ind + 1

        # now just fix the first link so it links to nothing
        links[0].parent_name = "None"
        return links

    @staticmethod
    def crossover(g1, g2):
        x1 = random.randint(0, len(g1)-1)
        x2 = random.randint(0, len(g2)-1)
        g3 = np.concatenate((g1[x1:], g2[x2:])) 
        if len(g3) > len(g1):
            g3 = g3[0:len(g1)] 
        return g3

    @staticmethod
    def point_mutate(genome, rate, amount):
        new_genome = copy.copy(genome)
        for gene in new_genome:
            for i in range(len(gene)):
                if random.random() < rate:
                    gene[i] += 0.1
                if gene[i] >= 1.0:
                    gene[i] = 0.9999
                if gene[i] < 0.0:
                    gene[i] = 0.0
        return new_genome

    @staticmethod
    def shrink_mutate(genome, rate):
        if len(genome) == 1:
            return copy.copy(genome)
        if random.random() < rate:
            ind = random.randint(0, len(genome)-1)
            new_genome = np.delete(genome, ind, 0)
            return new_genome
        else:
            return copy.copy(genome)

    @staticmethod
    def grow_mutate(genome, rate):
        if random.random() < rate:
            gene = Genome.get_random_gene(len(genome[0]))
            new_genome = copy.copy(genome)
            new_genome = np.append(new_genome, [gene], axis=0)
            return new_genome
        else:
            return copy.copy(genome)


    @staticmethod
    def to_csv(dna, csv_file):
        csv_str = ""
        for gene in dna:
            for val in gene:
                csv_str = csv_str + str(val) + ","
            csv_str = csv_str + '\n'

        with open(csv_file, 'w') as f:
            f.write(csv_str)

    @staticmethod
    def from_csv(filename):
        csv_str = ''
        with open(filename) as f:
            csv_str = f.read()   
        dna = []
        lines = csv_str.split('\n')
        for line in lines:
            vals = line.split(',')
            gene = [float(v) for v in vals if v != '']
            if len(gene) > 0:
                dna.append(gene)
        return dna

class URDFLink:
    def __init__(self, name, parent_name, recur, 
                link_length=0.1, 
                link_shape=0.1,
                link_radius=0.1, 
                link_mass=0.1,
                joint_type=0.1,
                joint_parent=0.1,
                joint_axis_xyz=0.1,
                joint_origin_rpy_1=0.1,
                joint_origin_rpy_2=0.1,
                joint_origin_rpy_3=0.1,
                joint_origin_xyz_1=0.1,
                joint_origin_xyz_2=0.1,
                joint_origin_xyz_3=0.1,
                control_waveform=0.1,
                control_amp=0.1,
                control_freq=0.1,
                control_phase=0.1,
                grip_strength=0.1,
                body_part_type=0.1,
                attachment_point=0.1,
                surface_friction=1.0,
                joint_friction=0.1):
        self.name = name
        self.parent_name = parent_name
        self.recur = recur 
        self.link_length=link_length 
        self.link_shape=link_shape
        self.link_radius=link_radius
        self.link_mass=link_mass
        self.joint_type=joint_type
        self.joint_parent=joint_parent
        self.joint_axis_xyz=joint_axis_xyz
        self.joint_origin_rpy_1=joint_origin_rpy_1
        self.joint_origin_rpy_2=joint_origin_rpy_2
        self.joint_origin_rpy_3=joint_origin_rpy_3
        self.joint_origin_xyz_1=joint_origin_xyz_1
        self.joint_origin_xyz_2=joint_origin_xyz_2
        self.joint_origin_xyz_3=joint_origin_xyz_3
        self.control_waveform=control_waveform
        self.control_amp=control_amp
        self.control_freq=control_freq
        self.control_phase=control_phase
        self.grip_strength=grip_strength
        self.body_part_type=body_part_type
        self.attachment_point=attachment_point
        self.surface_friction=surface_friction
        self.joint_friction=joint_friction
        self.sibling_ind = 1

    def to_link_element(self, adom):

        if self.link_shape < 0.33:
            shape_type = "cylinder"
        elif self.link_shape < 0.66:
            shape_type = "box"
        else:
            shape_type = "sphere"

        
  
        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", self.name)
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")
        
        if shape_type == "cylinder":
            shape_tag = adom.createElement("cylinder")
            shape_tag.setAttribute("length", str(self.link_length))
            shape_tag.setAttribute("radius", str(self.link_radius))
        elif shape_type == "box":
            shape_tag = adom.createElement("box")
            shape_tag.setAttribute("size", str(self.link_radius) + " " + str(self.link_radius) + " " + str(self.link_length))
        else:  # sphere
            shape_tag = adom.createElement("sphere")
            shape_tag.setAttribute("radius", str(self.link_radius))
        
        geom_tag.appendChild(shape_tag)
        vis_tag.appendChild(geom_tag)
        
        
        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")
        
        if shape_type == "cylinder":
            c_shape_tag = adom.createElement("cylinder")
            c_shape_tag.setAttribute("length", str(self.link_length))
            c_shape_tag.setAttribute("radius", str(self.link_radius))
        elif shape_type == "box":
            c_shape_tag = adom.createElement("box")
            c_shape_tag.setAttribute("size", str(self.link_radius) + " " + str(self.link_radius) + " " + str(self.link_length))
        else:  # sphere
            c_shape_tag = adom.createElement("sphere")
            c_shape_tag.setAttribute("radius", str(self.link_radius))
        
        c_geom_tag.appendChild(c_shape_tag)
        coll_tag.appendChild(c_geom_tag)
        
        # Add surface friction properties to collision
        surface_tag = adom.createElement("surface")
        friction_tag = adom.createElement("friction")
        ode_tag = adom.createElement("ode")
        mu_tag = adom.createElement("mu")
        mu_tag.appendChild(adom.createTextNode(str(self.surface_friction)))
        mu2_tag = adom.createElement("mu2")
        mu2_tag.appendChild(adom.createTextNode(str(self.surface_friction)))
        ode_tag.appendChild(mu_tag)
        ode_tag.appendChild(mu2_tag)
        friction_tag.appendChild(ode_tag)
        surface_tag.appendChild(friction_tag)
        coll_tag.appendChild(surface_tag)
        
        #     <inertial>
        # 	    <mass value="0.25"/>
        # 	    <inertia ixx="0.0003" iyy="0.0003" izz="0.0003" ixy="0" ixz="0" iyz="0"/>
        #     </inertial>
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        # pi r^2 * height
        mass = np.pi * (self.link_radius * self.link_radius) * self.link_length
        mass_tag.setAttribute("value", str(mass))
        inertia_tag = adom.createElement("inertia")
        # <inertia ixx="0.0003" iyy="0.0003" izz="0.0003" ixy="0" ixz="0" iyz="0"/>
        inertia_tag.setAttribute("ixx", "0.03")
        inertia_tag.setAttribute("iyy", "0.03")
        inertia_tag.setAttribute("izz", "0.03")
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyx", "0")
        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)
        

        link_tag.appendChild(vis_tag)
        link_tag.appendChild(coll_tag)
        link_tag.appendChild(inertial_tag)
        
        return link_tag

    def to_joint_element(self, adom):
        #           <joint name="base_to_sub2" type="revolute">
        #     <parent link="base_link"/>
        #     <child link="sub_link2"/>
        #     <axis xyz="1 0 0"/>
        #     <limit effort="10" upper="0" lower="10" velocity="1"/>
        #     <origin rpy="0 0 0" xyz="0 0.5 0"/>
        #   </joint>
        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", self.name + "_to_" + self.parent_name)
        if self.joint_type < 0.6:
            joint_tag.setAttribute("type", "revolute")
        elif self.joint_type < 0.66:
            joint_tag.setAttribute("type", "prismatic")
        else:
            joint_tag.setAttribute("type", "fixed")
        parent_tag = adom.createElement("parent")
        parent_tag.setAttribute("link", self.parent_name)
        child_tag = adom.createElement("child")
        child_tag.setAttribute("link", self.name)
        axis_tag = adom.createElement("axis")
        if self.joint_axis_xyz <= 0.33:
            axis_tag.setAttribute("xyz", "1 0 0")
        if self.joint_axis_xyz > 0.33 and self.joint_axis_xyz <= 0.66:
            axis_tag.setAttribute("xyz", "0 1 0")
        if self.joint_axis_xyz > 0.66:
            axis_tag.setAttribute("xyz", "0 0 1")
        
        limit_tag = adom.createElement("limit")
        # effort upper lower velocity
        limit_tag.setAttribute("effort", str(self.grip_strength))
        limit_tag.setAttribute("upper", "-3.1415")
        limit_tag.setAttribute("lower", "3.1415")
        limit_tag.setAttribute("velocity", "1")
        
        # Add joint friction for non-fixed joints
        if self.joint_type < 0.66:
            dynamics_tag = adom.createElement("dynamics")
            dynamics_tag.setAttribute("friction", str(self.joint_friction))
            joint_tag.appendChild(dynamics_tag)
        
        # <origin rpy="0 0 0" xyz="0 0.5 0"/>
        orig_tag = adom.createElement("origin")
        
        rpy1 = self.joint_origin_rpy_1 * self.sibling_ind
        rpy = str(rpy1) + " " + str(self.joint_origin_rpy_2) + " " + str(self.joint_origin_rpy_3)
        
        orig_tag.setAttribute("rpy", rpy)
        
        # Constrain joint positioning based on attachment point and body part type
        if self.attachment_point < 0.5:  # End attachment
            if self.body_part_type < 0.2:  # Torso extension
                xyz_1 = 0
                xyz_2 = 0  
                xyz_3 = self.joint_origin_xyz_3 * 0.3  # Small offset
            else:  # Limb/extremity from end
                xyz_1 = self.joint_origin_xyz_1 * 0.2
                xyz_2 = self.joint_origin_xyz_2 * 0.2
                xyz_3 = 0.5  # Attach at end
        else:  # Surface attachment
            if self.body_part_type >= 0.2 and self.body_part_type < 0.8:  # Limb from surface
                xyz_1 = (self.joint_origin_xyz_1 - 0.5) * 1.0  # Side attachment
                xyz_2 = (self.joint_origin_xyz_2 - 0.5) * 1.0
                xyz_3 = self.joint_origin_xyz_3 * 0.6
            else:  # Extremity from surface
                xyz_1 = (self.joint_origin_xyz_1 - 0.5) * 0.3
                xyz_2 = (self.joint_origin_xyz_2 - 0.5) * 0.3
                xyz_3 = (self.joint_origin_xyz_3 - 0.5) * 0.3
        
        xyz = str(xyz_1) + " " + str(xyz_2) + " " + str(xyz_3)
        orig_tag.setAttribute("xyz", xyz)

        joint_tag.appendChild(parent_tag)
        joint_tag.appendChild(child_tag)
        joint_tag.appendChild(axis_tag)
        joint_tag.appendChild(limit_tag)
        joint_tag.appendChild(orig_tag)
        return joint_tag