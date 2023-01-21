"""
This file provides the island model for metaoptimization. 
"""
import math
import os
import sys

from matplotlib import pyplot as plt
import networkx as nx

import argparse

from leap_ec import Individual, Representation, context, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import multi_population_ea
from leap_ec.real_rep.problems import SchwefelProblem, RastriginProblem, RosenbrockProblem, ShekelProblem
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.initializers import create_real_vector


##############################
# viz_plots function
##############################
def viz_plots(problems, modulo):
    """A convenience method that creates a figure with grid of subplots for
    visualizing the population genotypes and best-of-gen fitness for a number
    of different problems.

    :return: two lists of probe operators (for the phenotypes and fitness,
    respectively). Insert these into your algorithms to plot measurements to
    the respective subplots. """

    num_rows = min(4, len(problems))
    num_columns = math.ceil(len(problems) / num_rows)
    true_rows = len(problems) / num_columns
    
    # fig = plt.figure(figsize=(6 * num_columns, 2.5 * true_rows))
    # fig.tight_layout()
    
    genotype_probes = []
    fitness_probes = []
    for i, p in enumerate(problems):
        # plt.subplot(int(true_rows), int(num_columns) * 2, 2 * i + 1) # int() to ensure the values are integers
        tp = probe.CartesianPhenotypePlotProbe(
            contours=p,
            xlim=p.bounds,
            ylim=p.bounds,
            modulo=modulo,
            ax=plt.gca();
        )
        genotype_probes.append(tp)

        # plt.subplot(true_rows, num_columns * 2, 2 * i + 2)
        fp = probe.FitnessPlotProbe(ylim=(0, 1), modulo=modulo, 
        ax=plt.gca();
        )
        fitness_probes.append(fp)

    # plt.subplots_adjust(
    #     left=0.05,
    #     bottom=0.05,
    #     right=0.95,
    #     top=0.95,
    #     wspace=0.2,
    #     hspace=0.3)

    return genotype_probes, fitness_probes


def graph_drawer(topology):
    """
    Function for creating different graphs.
    """
    ...

##############################
# Entry point
##############################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This is a script designed for running Multipopulation Evolutionary Algorythms. ")

    parser.add_argument("-ps", "--population-size", type=int, default=15,\
        help="Defines the population size of a single island.")
    parser.add_argument("--selection",type=str, default="tournament", \
        help="defines selection type. Possible options: tournament,roulette,sus")
    parser.add_argument("--crossover", type=str, default="uniform",\
        help="uniform,multi_point,blend_alpha")
    parser.add_argument("--crossover-points", type=int, default=1,\
        help="")
    parser.add_argument("--mutation", type=str, default="shift_gaussian",\
        help="Possible options: shift_gaussian,replace_uniform")
    parser.add_argument("--populations", type=int, default=100,\
        help="Define total number of populations across all islands.")
    parser.add_argument("--interval", type=int, default=10,\
        help="")
    parser.add_argument("--topology", type=str, default="fully_connected",\
        help="Possible options: fully_connected,ring,mesh,star")
    parser.add_argument("--migration-selection", type=str, default="tournament",\
        help="Possible options: tournament,natural,roulette,sus")
    parser.add_argument("--immigration-selection", type=str, default="tournament",\
        help="Possible options: tournament,roulette,sus")
    
    args = parser.parse_args()

    print("Args:", args)

    pop_size = args.population_size

    # TOPOLOGY SELECTION:
    # Set up up the network of connections between islands

    topology = nx.complete_graph(pop_size)
    if args.topology == "star":
        if pop_size%2 == 0:
            topology = nx.star_graph(pop_size+1) #has to be an odd number, +1 for central island
        else:
            topology = nx.star_graph(pop_size)
    if args.topology == "ring":
        if pop_size%2 == 0:
            topology = nx.cycle_graph(pop_size)  # has to be an even number
        else:
            topology = nx.cycle_graph(pop_size+1)
    
    # mesh 2d and 3d can be produced as the same model, there will be a single 2d representation of it. 
    if args.topology == "mesh":
        if pop_size%2 == 0:
            n = pop_size+1  # has to be an odd number
        else:
            n = pop_size
        m = pop_size*2  # 2xpop_size edges
        topology = nx.gnm_random_graph(n, m, seed=42)
    
    nx.draw(topology)

    # DEFINE THE PROBLEM HERE:
    problem = ShekelProblem(maximize=False)

    # genotype_probes, fitness_probes = viz_plots([problem] * topology.number_of_nodes(), modulo=10)
    # subpop_probes = list(zip(genotype_probes, fitness_probes))

    def get_island(context):
        """Closure that returns a callback for retrieving the current island
        ID during logging."""
        return lambda _: context['leap']['current_subpopulation']
    
    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        # generations = 100
        generations = int(2 * args.populations / pop_size)
    l = 2
    
    # SELECTION:
    chosen_selection = ops.tournament_selection
    if args.selection == "sus":
        chosen_selection = ops.sus_selection
    if args.selection == "roulette":
        chosen_selection = ops.cyclic_selection

    # MIGRATION SELECTION:
    selected_migration = ops.tournament_selection
    if args.migration_selection == "sus":
        selected_migration = ops.sus_selection
    if args.migration_selection == "roulette":
        selected_migration = ops.cyclic_selection

    print("Generations:", generations, " | Populations:", args.populations, " | Population_size:", pop_size)

    ea = multi_population_ea(max_generations=generations,
                             num_populations=topology.number_of_nodes(),
                             pop_size=pop_size,
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 individual_cls=Individual,
                                 initialize=create_real_vector(
                                     bounds=[problem.bounds] * l)
                             ),

                             # Operator pipeline
                             shared_pipeline=[
                                 chosen_selection,
                                 ops.clone,
                                 mutate_gaussian(
                                     std=30,
                                     expected_num_mutations=1,
                                     hard_bounds=problem.bounds),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 ops.migrate(topology=topology,
                                             emigrant_selector=selected_migration,
                                             replacement_selector=ops.random_selection,
                                             migration_gap=50),
                                 probe.FitnessStatsCSVProbe(stream=sys.stdout,
                                        extra_metrics={ 'island': get_island(context) })
                            ],
                            # subpop_pipelines=subpop_probes
                            )

    list(ea)
