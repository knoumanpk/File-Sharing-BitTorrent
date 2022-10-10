from file_sharing import Context
from Network import Network

from itertools import groupby


# %-----------------------------------------------------------------------------------%
# %----------------Logging Functions-----------------------%
# %-----------------------------------------------------------------------------------%
class Logger:
    def __init__(self, network: Network, context: Context):
        # population logging
        self.network_population = {'time': [], 'population': []}
        self.network_empty_peers_count = {'time': [], 'empty_peers_count': [], 'population': []}
        self.swarm_population = {
            key: {'time': [], 'population': []} for key in network.swarm_container.keys()
        }
        self.swarm_empty_peers_count = {
            key: {'time': [], 'empty_peers_count': [], 'population': []} for key in network.swarm_container.keys()
        }
        # chunk_counts logging
        self.network_chunk_counts = {'time': [], 'population': []}
        for c in network.chunk_counts.keys():
            self.network_chunk_counts[c] = []
        self.swarm_chunk_counts = {}
        for key in network.swarm_container.keys():
            self.swarm_chunk_counts[key] = {
                'time': [],
                'population': [],
                'primary_file': network.swarm_container[key].primary_file
            }
            for c in network.swarm_container[key].chunk_counts.keys():
                self.swarm_chunk_counts[key][c] = []

    def log_event(self, event: tuple, network: Network, context: Context):
        event_name = event[0]
        if event_name == "peer_arrival":
            self.network_empty_peers_count['time'].append(context.time)
            self.network_empty_peers_count['empty_peers_count'].append(
                network.empty_peers_count
            )
            self.network_empty_peers_count['population'].append(
                network.total_population
            )
            self.network_population['time'].append(context.time)
            self.network_population['population'].append(
                network.total_population
            )
            # logging for corresponding swarm
            key = event[1]
            self.swarm_empty_peers_count[key]['time'].append(context.time)
            self.swarm_empty_peers_count[key]['empty_peers_count'].append(
                network.swarm_container[key].empty_peers_count
            )
            self.swarm_empty_peers_count[key]['population'].append(
                network.swarm_container[key].swarm_population
            )
            self.swarm_population[key]['time'].append(context.time)
            self.swarm_population[key]['population'].append(
                network.swarm_container[key].swarm_population
            )
        else:  # chunk_download
            swarm_keys = event[1]  # keys of swarms whose values to log
            keys = [key_iter[0] for key_iter in groupby(swarm_keys.values())]
            # above line removes one copy if both swarm_keys are the same

            # network_population
            if self.network_population['population'][-1] != network.total_population:
                self.network_population['time'].append(context.time)
                self.network_population['population'].append(
                    network.total_population
                )
            # network_empty_peers_count
            if self.network_empty_peers_count['empty_peers_count'][-1] != network.empty_peers_count \
                    or self.network_empty_peers_count['population'][-1] != network.total_population:
                self.network_empty_peers_count['time'].append(context.time)
                self.network_empty_peers_count['empty_peers_count'].append(
                    network.empty_peers_count
                )
                self.network_empty_peers_count['population'].append(
                    network.total_population
                )
            # network_chunk_counts
            if context.is_log_freq:
                self.network_chunk_counts['time'].append(context.time)
                self.network_chunk_counts['population'].append(
                    network.total_population
                )
                for c in self.network_chunk_counts.keys():
                    if c == 'time' or c == 'population':
                        continue
                    self.network_chunk_counts[c].append(
                        network.chunk_counts[c]
                    )
            # logging for corresponding swarms
            for key in keys:
                # swarm_population
                if self.swarm_population[key]['population'][-1] != network.swarm_container[key].swarm_population:
                    self.swarm_population[key]['time'].append(context.time)
                    self.swarm_population[key]['population'].append(
                        network.swarm_container[key].swarm_population
                    )
                # swarm empty_peers_count
                if self.swarm_empty_peers_count[key]['empty_peers_count'][-1] != network.swarm_container[key].empty_peers_count \
                        or self.swarm_empty_peers_count[key]['population'][-1] != network.swarm_container[key].swarm_population:
                    self.swarm_empty_peers_count[key]['time'].append(context.time)
                    self.swarm_empty_peers_count[key]['empty_peers_count'].append(
                        network.swarm_container[key].empty_peers_count
                    )
                    self.swarm_empty_peers_count[key]['population'].append(
                        network.swarm_container[key].swarm_population
                    )
                # swarm_chunk_counts
                if context.is_log_freq:
                    self.swarm_chunk_counts[key]['time'].append(context.time)
                    self.swarm_chunk_counts[key]['population'].append(
                        network.swarm_container[key].swarm_population
                    )
                    for c in self.swarm_chunk_counts[key].keys():
                        if c == 'time' or c == 'population' or c == 'primary_file':
                            continue
                        self.swarm_chunk_counts[key][c].append(
                            network.swarm_container[key].chunk_counts[c]
                        )
        return

    def log_all_histories(self, network: Network, context: Context):
        # network population
        self.network_population['time'].append(context.time)
        self.network_population['population'].append(network.total_population)
        # network empty_peers_count
        self.network_empty_peers_count['time'].append(context.time)
        self.network_empty_peers_count['empty_peers_count'].append(network.empty_peers_count)
        self.network_empty_peers_count['population'].append(network.total_population)
        # network chunk_counts
        if context.is_log_freq:
            self.network_chunk_counts['time'].append(context.time)
            self.network_chunk_counts['population'].append(network.total_population)
            for c in self.network_chunk_counts.keys():
                if c == 'time' or c == 'population':
                    continue
                self.network_chunk_counts[c].append(network.chunk_counts[c])
        # swarm populations and empty_peers_counts
        for key in self.swarm_population.keys():
            self.swarm_population[key]['time'].append(context.time)
            self.swarm_population[key]['population'].append(network.swarm_container[key].swarm_population)
        # swarm empty_peer_counts
        for key in self.swarm_empty_peers_count.keys():
            self.swarm_empty_peers_count[key]['time'].append(context.time)
            self.swarm_empty_peers_count[key]['empty_peers_count'].append(network.swarm_container[key].empty_peers_count)
            self.swarm_empty_peers_count[key]['population'].append(network.swarm_container[key].swarm_population)
        # swarm chunk_counts
        if context.is_log_freq:
            for key in self.swarm_chunk_counts.keys():
                self.swarm_chunk_counts[key]['time'].append(context.time)
                self.swarm_chunk_counts[key]['population'].append(network.swarm_container[key].swarm_population)
                for c in self.swarm_chunk_counts[key].keys():
                    if c == 'time' or c == 'population' or c == 'primary_file':
                        continue
                    self.swarm_chunk_counts[key][c].append(network.swarm_container[key].chunk_counts[c])
        return