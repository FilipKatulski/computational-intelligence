import networkx as nx
from leap_ec.algorithm import multi_population_ea
from leap_ec import ops
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep import problems
from leap_ec.decoder import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector

def main():

    topology = nx.complete_graph(4)
    nx.draw(topology)
    problem = problems.RastriginProblem(maximize=False)

    l = 2  # Length of the genome
    pop_size = 10
    ea = multi_population_ea(max_generations=1000,
                            num_populations=topology.number_of_nodes(),
                            pop_size=pop_size,

                            problem=problem,

                            representation=Representation(
                            # individual_cls=Individual,
                                decoder=IdentityDecoder(),
                                initialize=create_real_vector(bounds=[problem.bounds] * l)
                            ),

                            shared_pipeline=[
                                ops.tournament_selection,
                                ops.clone,
                                mutate_gaussian(std=30,
                                            expected_num_mutations='isotropic',
                                            hard_bounds=problem.bounds),
                                ops.evaluate,
                                ops.pool(size=pop_size),
                                ops.migrate(topology=topology,
                                            emigrant_selector=ops.tournament_selection,
                                            replacement_selector=ops.random_selection,
                                            migration_gap=50)
                            ])
    
    list(ea)


if __name__ == "__main__":
    main()  
