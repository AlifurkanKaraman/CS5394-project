"""
NEAT population orchestration for training.

TrainingManager loads the NEAT config file, creates a neat.Population, attaches
StdOutReporter and StatisticsReporter, then calls population.run(eval_fn, n_generations)
to drive the evolutionary training loop. Delegates per-generation evaluation to
SimulationEngine.
"""

pass
