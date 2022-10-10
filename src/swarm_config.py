import numpy as np

# %-----------------------------------------------------------------------------------%
# %----------------Default parameters for Single SwarmContainer-----------------------%
# %-----------------------------------------------------------------------------------%
single_swarm = dict({
        '1': {
            'swarm_key': '1',
            'arrival_rate': 0.,
            'primary_file': set(np.linspace(1, 100, 100, dtype=int)),
            'secondary_file': set(),
            'inter_swarm_behavior': 'altruistic',
            'ally_set': {'1'}
        }
    })

single_swarm['1']['alpha'] = 1e-9
single_swarm['1']['beta'] = 1.5
single_swarm['1']['TMS_threshold'] = 2 * len(single_swarm['1']['primary_file'])  # TMS
# single_swarm['1']['TMS_threshold'] = 0  # MS
single_swarm['1']['file'] = single_swarm['1']['primary_file'].union(single_swarm['1']['secondary_file'])


# %-----------------------------------------------------------------------------------%
# %----------------Default parameters for Single SwarmContainer-----------------------%
# %-----------------------------------------------------------------------------------%
two_swarm = dict({
        '1': {
            'swarm_key': '1',
            'arrival_rate': 0.,
            'primary_file': set(np.linspace(1, 100, 100, dtype=int)),
            'secondary_file': set(),
            'inter_swarm_behavior': 'altruistic',
            'ally_set': {'1'}
        },
        '2': {
'swarm_key': '1',
            'arrival_rate': 0.,
            'primary_file': set(np.linspace(1, 100, 100, dtype=int)),
            'secondary_file': set(),
            'inter_swarm_behavior': 'altruistic',
            'ally_set': {'1'}
        }

    })

two_swarm['1']['alpha'] = 1e-9
two_swarm['1']['beta'] = 1.5
two_swarm['1']['TMS_threshold'] = 2 * len(single_swarm['1']['primary_file'])  # TMS
# two_swarm['1']['TMS_threshold'] = 0  # MS
two_swarm['1']['file'] = two_swarm['1']['primary_file'].union(single_swarm['1']['secondary_file'])


# %-----------------------------------------------------------------------------------%
# %----------------Class Implementation of Swarm-----------------------%
# %-----------------------------------------------------------------------------------%
class Swarm:
    def __init__(self, config: dict = single_swarm['1']):
        """
        do nothing
        """
        self.swarm_key = config['swarm_key']
        self.arrival_rate = config['arrival_rate']
        self.file = config['file']
        self.primary_file = config['primary_file']
        self.secondary_file = config['secondary_file']
        self.inter_swarm_behavior = config['inter_swarm_behavior']
        self.ally_set = config['ally_set']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.TMS_threshold = config['TMS_threshold']

        self.empty_peers_count = 0
        self.swarm_population = 0
        self.chunk_counts = {chunk: int(0) for chunk in config['file']}

    def get_mode_count(self):
        """
        purpose: get the largest count from chunks in the primary file.
        """
        mode_count = np.max([self.chunk_counts[c] for c in self.primary_file])
        return mode_count

    def get_rarest_count(self):
        """
        purpose: get the smallest count from chunks in the primary file.
        """
        rarest_count = np.min([self.chunk_counts[c] for c in self.primary_file])
        return rarest_count

    def get_greatest_mismatch(self):
        """
        purpose: return largest mismatch (computed over pieces in the primary file).
        """
        return self.get_mode_count() - self.get_rarest_count()

    def get_set_of_rare_pieces(self):
        """
        purpose: returns the set of rare-pieces.
        """
        mode_count = self.get_mode_count()
        rarest_count = self.get_rarest_count()
        if mode_count == rarest_count:
            return self.primary_file
        else:
            return set([c for c in self.chunk_counts.keys() if (
                        c in self.primary_file and self.chunk_counts[c] < mode_count)])

    def get_rarest_pieces(self, available_pieces: set):
        """
        purpose: return the set of pieces with the lowest chunk_count from the given set.
        """
        rarest_count = min([self.chunk_counts[c] for c in available_pieces])
        return set([c for c in available_pieces if self.chunk_counts[c] == rarest_count])


# %-----------------------------------------------------------------------------------%
# %----------------Class Implementation of SwarmContainer-----------------------%
# %-----------------------------------------------------------------------------------%
class SwarmContainer(dict):
    def __init__(self, swarms_config: dict = single_swarm):
        """
        do nothing
        """
        super().__init__()
        for key, config in swarms_config.items():
            self[key] = Swarm(config=config)



