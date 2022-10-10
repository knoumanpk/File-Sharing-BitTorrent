import numpy as np
import math
from utils import exchange_chunks, compute_A_trans, download_from_A_trans


# %-----------------------------------------------------------------------------------%
# %----------------Class Implementation of Network-----------------------%
# %-----------------------------------------------------------------------------------%
class Network(list):
    def __init__(self, context, master_file: set, swarm_container):
        super().__init__()
        self.master_file = master_file
        self.swarm_container = swarm_container

        self.total_population = 0
        self.empty_peers_count = 0
        self.chunk_counts = {c: int(0) for c in master_file}

    def set_initial_condition(self, initial_condition: str = "flash_crowd"):
        """
        purpose: set initial condition of the network.
        @param initial_condition:
        @return:
        """
        if initial_condition == "flash_crowd":
            n_0 = 500
            for key in self.swarm_container.keys():
                for i in range(n_0):
                    self.append(
                        Peer(swarm=self.swarm_container[key], cache=set(),
                             entry_time=0.)
                    )
                    self.empty_peers_count += 1
                    self.swarm_container[key].empty_peers_count += 1
                    self.total_population += 1
                    self.swarm_container[key].swarm_population += 1
        elif initial_condition == "one_club":
            n_0 = 500
            for key in self.swarm_container.keys():
                primary_file = self.swarm_container[key].primary_file
                missing_piece = list(primary_file)[0]
                for i in range(n_0):
                    self.append(
                        Peer(swarm=self.swarm_container[key], cache=primary_file.difference({missing_piece}),
                             entry_time=0.)
                    )
                    self.total_population += 1
                    self.swarm_container[key].swarm_population += 1
                    for c in primary_file.difference({missing_piece}):
                        self.swarm_container[key].chunk_counts[c] += 1
                        self.chunk_counts[c] += 1
        else:
            pass
        return


# %-----------------------------------------------------------------------------------%
# %----------------Class Implementation of Seed-----------------------%
# %-----------------------------------------------------------------------------------%
class Seed:
    def __init__(self, network: Network):
        self.cache = network.master_file
        self.ally_set = set(network.swarm_container.keys())

    def optimistic_unchoke(self, network: Network, context, logger):
        """
        purpose: contact a normal peer via an optimistic-unchoke link.
        @param network: Network to which seed belongs.
        @param context: Context to carry across functions.
        @param logger: Logger to log histories.
        @return:
        """
        contacted_peer = self.make_contact(network=network, context=context)
        if contacted_peer is None:
            return

        # compute A_trans for contacted peer
        A_trans = compute_A_trans(
            revealed_cache=self.cache,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )

        # chunk-download by contacted peer from A_trans (if non-empty)
        download_from_A_trans(
            peer={1: contacted_peer},
            A_trans={1: A_trans},
            network=network,
            context=context,
            logger=logger
        )
        return

    def make_contact(self, network: Network, context):
        """
        purpose: make contact with a normal peer.
        @param network: Network to which the seed belongs.
        @param context: Context to pass values around functions.
        @return: None or contacted_peer (type: Peer)
        """
        if context.is_autonomous:
            # below line is based on dividing seed's throughput equally across all swarms
            swarm_key = np.random.choice([key for key in network.swarm_container.keys()])
            # return if swarm population is zero
            if network.swarm_container[swarm_key].swarm_population == 0:
                return None
            # otherwise choose one peer randomly from the swarm
            else:
                contacted_peer = np.random.choice([peer for peer in network if peer.swarm_key == swarm_key])
        else:
            # return if network population is zero
            if network.total_population == 0:
                return None
            # otherwise choose one peer randomly from the network
            else:
                contacted_peer = np.random.choice([peer for peer in network])
        return contacted_peer


# %-----------------------------------------------------------------------------------%
# %----------------Class Implementation of Peer-----------------------%
# %-----------------------------------------------------------------------------------%
class Peer:
    def __init__(self, swarm, cache: set, entry_time: float = 0.,
                 exit_time: float = math.inf):
        self.swarm_key = swarm.swarm_key
        self.swarm = swarm
        self.primary_file = swarm.primary_file
        self.secondary_file = swarm.secondary_file
        self.file = swarm.file
        self.cache = cache
        self.ally_set = swarm.ally_set
        self.entry_time = float(entry_time)
        self.exit_time = float(exit_time)
        # self.is_tit_for_tat = False
        # self.is_opt_unchoke = True

    def get_sjn_time(self):
        """
        purpose: return sojourn time of the peer.
        """
        return float(self.exit_time - self.entry_time)

    def make_contact(self, network: Network, context):
        """
        purpose: make contact with another normal peer.
        @param network: Network to which the peer belongs.
        @param context: Context to pass values across functions.
        @return: None or contacted_peer (type: Peer)
        """
        if context.is_autonomous:
            # return None if swarm population is less than 2
            if network.swarm_container[self.swarm_key].swarm_population < 2:
                return None
            # otherwise choose another peer randomly from the swarm
            else:
                contacted_peer = np.random.choice(
                    [peer for peer in network if peer.swarm_key == self.swarm_key and peer != self]
                )
        else:
            # return None if network population is less than 2
            if network.total_population < 2:
                return None
            # otherwise contact another peer randomly from the network
            else:
                contacted_peer = np.random.choice([peer for peer in network if peer != self])
        return contacted_peer

    def tit_for_tat_exchange(self, network: Network, context, logger):
        """
        purpose: contact another normal peer via a tit-for-tat link.
        @param network: Network to which the peer belongs.
        @param context: Context to pass across functions.
        @param logger: Logger to log histories.
        """
        # contact another (normal) peer
        contacted_peer = self.make_contact(network=network, context=context)
        if contacted_peer is None:
            return

        # the two peers exchange chunks based on tit-for-tat mechanism
        exchange_chunks(
            peer={1: self, 2: contacted_peer},
            network=network,
            context=context,
            logger=logger
        )
        return

    def optimistic_unchoke(self, network: Network, context, logger):
        """
        purpose: contact another normal peer via the optimistic-unchoke link.
        @param network: Network to which the peer belongs.
        @param context: Context to pass values across functions.
        @param logger: Logger to log histories.
        """
        # contact another (normal) peer
        contacted_peer = self.make_contact(network=network, context=context)
        if contacted_peer is None:
            return

        # revealed cache profile based on whether contacted peer is an ally
        S_hat = self.cache if contacted_peer.swarm_key in self.ally_set else set()

        # compute A_trans for contacted peer when revealed cache = S_hat
        A_trans = compute_A_trans(
            revealed_cache=S_hat,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )

        # chunk-download by contacted peer from A_trans (if non-empty)
        download_from_A_trans(
            peer={1: contacted_peer},
            A_trans={1: A_trans},
            network=network,
            context=context,
            logger=logger
        )
        return
