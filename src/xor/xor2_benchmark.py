""" Benchmark performance of 2-input XOR example. """
from __future__ import print_function

import cProfile
import math
import os
import pstats

from neat import nn
from neat import population
from neat.config import Config
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


pr = cProfile.Profile()
pr.enable()
bench(500)
pr.disable()
ps = pstats.Stats(pr).sort_stats('time')
ps.print_stats()
