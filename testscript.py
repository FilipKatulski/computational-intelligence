# Test script to check capabiliteis of leap_ec 

from leap_ec import ops 
from leap_ec.algorithm import generational_ea
from leap_ec.decoder import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
pop_size = 5
ea = generational_ea(max_generations=100, pop_size=pop_size,
                    problem=MaxOnes(),             # Solve a MaxOnes Boolean optimization problem

                    representation=Representation(
                        decoder=IdentityDecoder(),             # Genotype and phenotype are the same for this task
                        initialize=create_binary_sequence(length=10)  # Initial genomes are random binary sequences
                    ),

                    # The operator pipeline
                    pipeline=[ops.tournament_selection,                     # Select parents via tournament_selection selection
                            ops.clone,                          # Copy them (just to be safe)
                                mutate_bitflip,                 # Basic mutation: defaults to a 1/L mutation rate
                            ops.uniform_crossover(p_swap=0.4),  # Crossover with a 40% chance of swapping each gene
                            ops.evaluate,                       # Evaluate fitness
                            ops.pool(size=pop_size)             # Collect offspring into a new population
                    ])

print('Generation, Best_Individual')
for i, best in ea:
    print(f"{i}, {best}")