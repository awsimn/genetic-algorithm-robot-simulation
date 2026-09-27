import unittest
import population
import simulation 
import genome 
import creature 
import numpy as np
import multiprocessing as mp
import pybullet as p
import csv
import gc
import random

def evaluate_creature(cr):
    sim = simulation.Simulation(p.DIRECT)
    sim.run_creature(cr, 2600)
    cr.final_fitness = cr.get_fitness()
    p.disconnect(sim.physicsClientId)
    return cr

def tournament_selection(creatures, k=3):
    # Randomly select 'k' creatures from the population
    candidates = random.sample(creatures, k)
    
    # Choose one with the highest final fitness among the candidates
    return max(candidates, key=lambda cr: cr.final_fitness)

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=50, gene_count=6)

        log_file = "fitness_function1_log.csv"
        log_f = open(log_file, "w", newline="")
        writer = csv.writer(log_f)
        writer.writerow(["iteration", "fittest", "mean_fitness", "mean_links", "max_links"])

        # To check if fitness has stagnated over the past 10 generations
        last_best = None
        stagnation_counter = 0
        # Amount of top elites to pass unmutated onto next generation
        elite_count = 2


        for iteration in range(1001):
            # Use max amount of cores(or cores * 2 if hyperthreading supported) to parallelize population sim workload
            with mp.Pool(processes=mp.cpu_count()) as pool:
                # Assign 2 creatures at a time to a worker process
                pop.creatures = pool.map(evaluate_creature, pop.creatures, chunksize=2)

            fits = [cr.final_fitness for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]


            fittest = np.round(np.max(fits), 3)
            mean_fit = np.round(np.mean(fits), 3)
            mean_links = np.round(np.mean(links), 1)
            max_links = int(np.max(links))

            print(iteration, "fittest:", fittest,
                  "mean:", mean_fit, "mean links", mean_links, "max links", max_links)

            writer.writerow([iteration, fittest, mean_fit, mean_links, max_links])
            log_f.flush()

            # If fitness same as last generations update stagnation counter
            if last_best is not None and fittest <= last_best:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best = fittest

            # Adaptive mutation rates
            if stagnation_counter > 10:
                print("Stagnation detected. Increasing mutation rates.")
                point_rate = 0.3
                grow_rate = 0.25
                shrink_rate = 0.2
            else:
                point_rate = 0.1
                grow_rate = 0.1
                shrink_rate = 0.08

            new_creatures = []
            for _ in range(len(pop.creatures)):
                # Select parents through tournaments
                p1 = tournament_selection(pop.creatures, k=5)
                p2 = tournament_selection(pop.creatures, k=5)

                # Crossover between parents and perform random mutations
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=point_rate, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
                dna = genome.Genome.grow_mutate(dna, rate=grow_rate)

                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)

            # Top-2 elitism
            top_indices = np.argsort(fits)[-elite_count:]
            for i, idx in enumerate(top_indices):
                cr = pop.creatures[idx]
                elite = creature.Creature(1)
                elite.update_dna(cr.dna)
                elite.final_fitness = cr.final_fitness
                new_creatures[i] = elite

                if iteration % 50 == 0 and i<2:
                    filename = f"elite_{iteration}_rank{i+1}.csv"
                    genome.Genome.to_csv(cr.dna, filename)

            # Inject 2 randoms for diversity
            for i in range(2):
                new_creatures[-(i+1)] = creature.Creature(gene_count=6)

            pop.creatures = new_creatures
            gc.collect()

        log_f.close()
        self.assertNotEqual(fits[0], 0)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    unittest.main()
