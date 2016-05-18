""" Benchmark performance of 2-input XOR training """
from __future__ import print_function
import math
import os
import copy
from random import random, gauss, choice, randint
from multiprocessing import Pool
import pprint
# import matplotlib.pyplot as plt

from neat import population
from neat.config import Config
from neat import nn
from neat.math_util import mean, stdev

PROFILE = True

# XOR-2
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUTS = [0, 1, 1, 0]


def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)

        error = 0.0
        for i, inputs in enumerate(INPUTS):
            # serial activation
            output = net.serial_activate(inputs)
            error += (output[0] - OUTPUTS[i]) ** 2

        g.fitness = 1 - math.sqrt(error / len(OUTPUTS))


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'xor2_config'))
    config.report = False

    pop = population.Population(config)
    pop.run(eval_fitness, 1000)

    winner = pop.statistics.best_genome()

    num_hidden = 0
    for ng in winner.node_genes.values():
        if ng.type == 'HIDDEN':
            num_hidden += 1

    num_connections = len(winner.conn_genes)
    num_enabled = 0
    for cg in winner.conn_genes.values():
        if cg.enabled:
            num_enabled += 1

    return pop.generation, num_hidden, num_connections, num_enabled


def bench(num_runs):
    results = []
    while len(results) < num_runs:
        try:
            result = run()
        except population.CompleteExtinctionException:
            continue

        results.append(result)
        if len(results) % 10 == 0:
            print("Completed run %d of %d" % (len(results), num_runs))

    generations = [r[0] for r in results]
    hidden = [r[1] for r in results]
    connections = [r[2] for r in results]
    enabled = [r[3] for r in results]

    print("              mean (stdev)")
    print(" Generations: %.3f (%.3f)" % (mean(generations), stdev(generations)))
    print("hidden nodes: %.3f (%.3f)" % (mean(hidden), stdev(hidden)))
    print(" connections: %.3f (%.3f)" % (mean(connections), stdev(connections)))
    print("     enabled: %.3f (%.3f)" % (mean(enabled), stdev(enabled)))

    # plt.figure()
    # plt.title("Generations")
    # plt.hist(results[0], bins=50)
    #
    # plt.figure()
    # plt.title("Hidden node count")
    # plt.hist(results[1], bins=50)
    #
    # plt.figure()
    # plt.title("Total connection count")
    # plt.hist(results[2], bins=50)
    #
    # plt.figure()
    # plt.title("Enabled connection count")
    # plt.hist(results[3], bins=50)
    #
    # plt.show()


class GASimple(object):
    def __init__(self, population_size, genome_bounds, shift_mutate, replace_mutate, crossover):
        self.population_size = population_size
        self.genome_bounds = genome_bounds
        self.shift_mutate = shift_mutate
        self.replace_mutate = replace_mutate
        self.crossover = crossover
        self.num_workers = 6
        self.carryover_fraction = 0.5
        self.best_genome = None
        self.current_genomes = []

    def new_genome(self):
        return [b[0] + random() * (b[1] - b[0]) for b in self.genome_bounds]

    def run(self, num_generations, eval_func, goal, parallel=True):
        best_score = -1e38
        full_cutoff = int(math.ceil(self.population_size * self.carryover_fraction))

        for i in range(num_generations):
            fitness = []
            if parallel:
                # Compute fitness for genomes in parallel.
                pool = Pool(self.num_workers)
                jobs = []
                for g in self.current_genomes:
                    jobs.append(pool.apply_async(eval_func, (g,)))

                for job, g in zip(jobs, self.current_genomes):
                    f = job.get()
                    fitness.append((f, g))
                    if f > best_score:
                        best_score = f
                        self.best_genome = g

                pool.terminate()
                pool.close()
            else:
                for g in self.current_genomes:
                    f = eval_func(g)
                    fitness.append((f, g))
                    if f > best_score:
                        best_score = f
                        self.best_genome = g

            if best_score >= goal:
                break

            fitness.sort(reverse=True)
            cutoff_idx = min(len(fitness) - 1, max(1, full_cutoff))
            cutoff_fitness = fitness[cutoff_idx][0]

            print("%d fitness %.3f/%.3f/%.3f" % (i, fitness[0][0], cutoff_fitness, fitness[-1][0]))
            for f, g in fitness[:cutoff_idx]:
                print("  %.3f %r" % (f, g))

            print("best: %f" % best_score)
            print(",".join("%.3f" % q for q in self.best_genome))

            old_genomes = [g for f, g in fitness[:cutoff_idx + 1]]
            self.current_genomes = old_genomes

            # Replace low-fitness members.
            while len(self.current_genomes) < self.population_size:
                if random() < self.crossover and len(old_genomes) > 1:
                    # Crossover.
                    parent1 = choice(old_genomes)
                    parent2 = choice(old_genomes)
                    self.current_genomes.append([choice([a, b]) for a, b in zip(parent1, parent2)])
                else:
                    # Clone.
                    self.current_genomes.append(copy.deepcopy(choice(self.current_genomes)))

            # Mutate new population.
            for g in self.current_genomes:
                for j, (b0, b1) in enumerate(self.genome_bounds):
                    r = random()
                    if r < self.shift_mutate + self.replace_mutate:
                        if r < self.shift_mutate:
                            g[j] = max(b0, min(b1, g[j] + gauss(0.0, 0.01 * (b1 - b0))))
                        else:
                            g[j] = b0 + random() * (b1 - b0)


def optimize_generations(num_runs):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'xor2_config'))
    config.report = False

    x0 = [config.prob_add_conn,
          config.prob_add_node,
          config.prob_delete_conn,
          config.prob_delete_node,
          config.prob_mutate_bias,
          config.bias_mutation_power,
          config.prob_mutate_response,
          config.response_mutation_power,
          config.prob_mutate_weight,
          config.prob_replace_weight,
          config.weight_mutation_power,
          config.prob_toggle_link]

    bounds = [(0.01, 0.99),
              (0.01, 0.99),
              (0.01, 0.99),
              (0.01, 0.99),
              (0.01, 0.99),
              (0.01, 5.0),
              (0.01, 0.99),
              (0.01, 5.0),
              (0.01, 0.99),
              (0.01, 0.99),
              (0.01, 5.0),
              (0.01, 0.99)]

    def F(x):
        c = copy.deepcopy(config)
        c.prob_add_conn = x[0]
        c.prob_add_node = x[1]
        c.prob_delete_conn = x[2]
        c.prob_delete_node = x[3]
        c.prob_mutate_bias = x[4]
        c.bias_mutation_power = x[5]
        c.prob_mutate_response = x[6]
        c.response_mutation_power = x[7]
        c.prob_mutate_weight = x[8]
        c.prob_replace_weight = x[9]
        c.weight_mutation_power = x[10]
        c.prob_toggle_link = x[11]

        generations = []
        hidden = []
        while len(generations) < num_runs:
            pop = population.Population(c)
            try:
                pop.run(eval_fitness, 80)
            except population.CompleteExtinctionException:
                continue

            generations.append(pop.generation)
            # print(pop.generation)

            h = 0
            for ng in pop.best_genome().node_genes.values():
                if ng.type == 'HIDDEN':
                    h += 1
            hidden.append(h)

        m = sum(generations) / float(num_runs)
        h = sum(hidden) / float(num_runs)
        score = ((m - 10) / 10.0) ** 2 + (h - 1) ** 2
        print(",".join("%.3f" % q for q in x) + " " + str(m) + " " + str(h) + " " + str(score))

        return -score

    ga = GASimple(6 * 8, bounds, 0.9, 0.1, 0.7)
    ga.current_genomes.append(x0)
    ga.run(100, F, 0.0)

    # xopt = minimize(F, x0)
    # xopt = differential_evolution(F, bounds)
    print(ga.best_genome)


def bench_distance():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'xor2_config'))
    config.report = False

    pop = population.Population(config)
    total_pop = []
    for s in pop.species:
        total_pop.extend(s.members)

    for a in total_pop:
        for b in total_pop:
            a.distance(b)





if __name__ == '__main__':
    if not PROFILE:
        #optimize_generations(50)
        bench(5000)
    else:
        import cProfile, pstats, StringIO

        pr = cProfile.Profile()
        pr.enable()
        bench(500)
        #bench_distance()
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'time'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
