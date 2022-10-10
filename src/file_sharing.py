import numpy as np
import os
import time
from datetime import datetime
import json
import random
import argparse
from tqdm import tqdm

from swarm_config import single_swarm, SwarmContainer
from Network import Network, Peer, Seed


# %-----------------------------------------------------------------------------------%
# %----------------Seed Function-----------------------%
# %-----------------------------------------------------------------------------------%
def same_seeds(seed: int = 1):
    """
    :purpose: set seed to given value for reproducibility
    @param seed: value of seed. (type: int)
    """
    np.random.seed(seed)  # numpy random module.
    random.seed(seed)  # python random module.
    return


# %-----------------------------------------------------------------------------------%
# %----------------Simple Class to carry variables to each trainer process-----------------------%
# %-----------------------------------------------------------------------------------%
class Context:
    """
    :purpose: Class to carry all important variables.
    """

    def __init__(self):
        """
        do nothing.
        """
        pass


# %-----------------------------------------------------------------------------------%
# %----------------Function for Next Event-----------------------%
# %-----------------------------------------------------------------------------------%
def sample_next_event(network: Network, context: Context):
    """
    purpose: determines the next event along with required information.
    @param network: Network being simulated.
    @param context: Context to carry across functions.
    @return: next_event: (type: tuple)
     - first entry of the tuple is name of the event.
     - rest of the entries contain info about the named event.
    """
    peer_arrival_times = np.random.exponential(
        [(1 / network.swarm_container[str(k)].arrival_rate) if network.swarm_container[str(k)].arrival_rate else np.inf
         for k in range(1, len(network.swarm_container) + 1)]
    )

    seed_contact_times = np.random.exponential([(1 / context.U)] * context.L)

    if context.Y_opt:
        temp_1 = np.ones((network.total_population, context.L - 1)) * (1 / context.mu)
        temp_2 = np.ones((network.total_population, 1)) * (1 / context.mu_hat)
        peer_tick_parameters = np.concatenate((temp_1, temp_2), axis=1)
    else:
        peer_tick_parameters = np.ones((network.total_population, context.L)) * (1 / context.mu)
    peer_contact_links_times = np.random.exponential(peer_tick_parameters) \
        if network.total_population else np.array([np.inf])

    if np.min(peer_arrival_times) <= min(np.min(seed_contact_times), np.min(peer_contact_links_times)):
        time_step = np.min(peer_arrival_times)
        swarm_key = str(np.argmin(peer_arrival_times) + 1)
        next_event = (
            "peer_arrival",
            swarm_key,
            time_step
        )
    elif np.min(seed_contact_times) <= min(np.min(peer_arrival_times), np.min(peer_contact_links_times)):
        time_step = np.min(seed_contact_times)
        # link_index = np.argmin(seed_contact_times)
        next_event = (
            "seed_contact",
            "opt_unchoke",
            time_step
        )
    else:
        time_step = np.min(peer_contact_links_times)
        peer_index, link_index = np.unravel_index(np.argmin(peer_contact_links_times), peer_contact_links_times.shape)
        next_event = (
            "peer_contact",
            peer_index,
            "opt_unchoke" if (link_index == (context.L - 1) and context.Y_opt) else "tit_for_tat",
            time_step
        )
    return next_event


# %-----------------------------------------------------------------------------------%
# %----------------Function for setting inter-swarm behavior-----------------------%
# %-----------------------------------------------------------------------------------%
def set_inter_swarm_behavior(swarms_config: dict, master_file: set):
    """
    purpose: set the file, secondary_file, and ally_set of each swarm in swarms-configuration.
    @param swarms_config: dict
    @param master_file: Network's master-file {type: set}
    @return:
    """
    for key in swarms_config:
        # configure file, secondary-file, ally_set based on behavior
        if swarms_config[key]['inter_swarm_behavior'] == "altruistic":
            swarms_config[key]['file'] = master_file
            swarms_config[key]['secondary_file'] = swarms_config[key]['file'].difference(
                swarms_config[key]['primary_file']
            )
            swarms_config[key]['ally_set'] = set(swarms_config.keys())
        elif swarms_config[key]['inter_swarm_behavior'] == "opportunistic":
            swarms_config[key]['file'] = swarms_config[key]['primary_file']
            swarms_config[key]['secondary_file'] = set()
            swarms_config[key]['ally_set'] = set(swarms_config.keys())
        elif swarms_config[key]['inter_swarm_behavior'] == "selfish":
            swarms_config[key]['file'] = swarms_config[key]['primary_file']
            swarms_config[key]['secondary_file'] = set()
            swarms_config[key]['ally_set'] = {key}
        else:  # autonomous
            swarms_config[key]['file'] = swarms_config[key]['primary_file']
            swarms_config[key]['secondary_file'] = set()
            swarms_config[key]['ally_set'] = {key}
    return swarms_config


# %-----------------------------------------------------------------------------------%
# %----------------Stability Check-----------------------%
# %-----------------------------------------------------------------------------------%
def stability_check():
    """
    purpose: check stability of SB-RFwPMS.
    @return:
    """
    arrival_rates = [(20., 20.)]
    file_configs = [
        (set(np.linspace(1, 15, 15, dtype=int)), set(np.linspace(11, 25, 15, dtype=int)))
    ]
    behaviors = ["altruistic", "opportunistic", "selfish", "autonomous"]
    for behavior in behaviors:
        for arrival_rate_1, arrival_rate_2 in arrival_rates:
            for W_1, W_2 in file_configs:
                master_file = W_1.union(W_2)
                swarms_config = {
                    '1': {
                        'swarm_key': '1',
                        'arrival_rate': arrival_rate_1,
                        'primary_file': W_1,
                        'inter_swarm_behavior': behavior,
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W_1)
                    },
                    '2': {
                        'swarm_key': '2',
                        'arrival_rate': arrival_rate_2,
                        'primary_file': W_2,
                        'inter_swarm_behavior': behavior,
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W_2)
                    }
                }
                swarms_config = set_inter_swarm_behavior(swarms_config=swarms_config, master_file=master_file)
                args = get_parser().parse_args(
                    [
                        "--results_directory", os.path.join("./../results/stability/", behavior),
                        "--U", '1.',
                        "--mu", '1.',
                        "--mu_hat", str(1/3),
                        "--L", '3',
                        "--p", '0.',
                        "--initial_condition", "one_club",
                        "--policy", "SB-RFwPMS",
                        "--end_time", '300.',
                        "--seed", '1',
                    ]
                )
                args.master_file = master_file
                args.swarms_config = swarms_config
                args.is_autonomous = True if behavior == "autonomous" else False
                args.Y_opt = True
                args.is_log_freq = True
                main(args)

    return


# %-----------------------------------------------------------------------------------%
# %----------------Scalability Check-----------------------%
# %-----------------------------------------------------------------------------------%
def scalability_check():
    """
    purpose: check scalability of SB-RFwPMS.
    @return:
    """
    base_arrival_rate_vector = (4., 2.)
    arrival_rates = [
        base_arrival_rate_vector,
        tuple(4 * j for j in base_arrival_rate_vector),
        tuple(16 * j for j in base_arrival_rate_vector)
    ]
    file_configs = [
        (set(np.linspace(1, 10, 10, dtype=int)), set(np.linspace(9, 18, 10, dtype=int)))
    ]
    behaviors = ["altruistic", "opportunistic", "selfish", "autonomous"]
    for behavior in behaviors:
        for arrival_rate_1, arrival_rate_2 in arrival_rates:
            for W_1, W_2 in file_configs:
                master_file = W_1.union(W_2)
                swarms_config = {
                    '1': {
                        'swarm_key': '1',
                        'arrival_rate': arrival_rate_1,
                        'primary_file': W_1,
                        'inter_swarm_behavior': behavior,
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W_1)
                    },
                    '2': {
                        'swarm_key': '2',
                        'arrival_rate': arrival_rate_2,
                        'primary_file': W_2,
                        'inter_swarm_behavior': behavior,
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W_2)
                    }
                }
                swarms_config = set_inter_swarm_behavior(swarms_config=swarms_config, master_file=master_file)
                args = get_parser().parse_args(
                    [
                        "--results_directory", os.path.join("./../results/scalability/", behavior),
                        "--U", '1.',
                        "--mu", '1.',
                        "--mu_hat", str(1/3),
                        "--L", '5',
                        "--p", '0.5',
                        "--initial_condition", "zero",
                        "--policy", "SB-RFwPMS",
                        "--end_time", '1000.',
                        "--seed", '1',
                    ]
                )
                args.master_file = master_file
                args.swarms_config = swarms_config
                args.is_autonomous = True if behavior == "autonomous" else False
                args.Y_opt = True
                args.is_log_freq = True
                main(args)
    return


# %-----------------------------------------------------------------------------------%
# %----------------Sojourn Time Performance-----------------------%
# %-----------------------------------------------------------------------------------%
def sojourn_time_performance():
    """
    purpose: evaluate sojourn time performance of SB-RFwPMS.
    @return:
    """
    arrival_rates = [(4., 1.)]
    file_configs = [
        (set(np.linspace(1, 2, 2, dtype=int)), set(np.linspace(2, 3, 2, dtype=int))),
        (set(np.linspace(1, 10, 10, dtype=int)), set(np.linspace(6, 15, 10, dtype=int))),
        (set(np.linspace(1, 20, 20, dtype=int)), set(np.linspace(11, 30, 20, dtype=int))),
        (set(np.linspace(1, 40, 40, dtype=int)), set(np.linspace(21, 60, 40, dtype=int))),
        (set(np.linspace(1, 80, 80, dtype=int)), set(np.linspace(41, 120, 80, dtype=int))),
        (set(np.linspace(1, 100, 100, dtype=int)), set(np.linspace(51, 150, 100, dtype=int))),
        # (set(np.linspace(1, 200, 200, dtype=int)), set(np.linspace(101, 300, 200, dtype=int))),
        # (set(np.linspace(1, 500, 500, dtype=int)), set(np.linspace(251, 750, 500, dtype=int)))
    ]
    behaviors = ["altruistic", "opportunistic", "selfish", "autonomous"]
    for behavior in behaviors:
        for arrival_rate_1, arrival_rate_2 in arrival_rates:
            for W_1, W_2 in file_configs:
                master_file = W_1.union(W_2)
                swarms_config = {
                    '1': {
                        'swarm_key': '1',
                        'arrival_rate': arrival_rate_1,
                        'primary_file': W_1,
                        'inter_swarm_behavior': behavior,
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W_1)
                    },
                    '2': {
                        'swarm_key': '2',
                        'arrival_rate': arrival_rate_2,
                        'primary_file': W_2,
                        'inter_swarm_behavior': behavior,
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W_2)
                    }
                }
                swarms_config = set_inter_swarm_behavior(swarms_config=swarms_config, master_file=master_file)
                args = get_parser().parse_args(
                    [
                        "--results_directory", os.path.join(
                        "./../results/sojourn_time_performance/", behavior, "KW_1_"+str(len(W_1))+"_KW_2_"+str(len(W_2))
                    ),
                        "--U", '1.',
                        "--mu", '1.',
                        "--mu_hat", str(1/3),
                        "--L", '5',
                        "--p", '0.5',
                        "--initial_condition", "zero",
                        "--policy", "SB-RFwPMS",
                        "--end_time", '5000.',
                        "--seed", '1',
                    ]
                )
                args.master_file = master_file
                args.swarms_config = swarms_config
                args.is_autonomous = True if behavior == "autonomous" else False
                args.Y_opt = False
                args.is_log_freq = True if max(len(W_1), len(W_2)) <= 10 else False
                main(args)
    return


# %-----------------------------------------------------------------------------------%
# %----------------Single Swarm Sojourn Times-----------------------%
# %-----------------------------------------------------------------------------------%
def single_swarm_sojourn_times():
    """
    purpose: evaluate sojourn times in single-swarm for SB-RFwPMS and SB-MS.
    @return:
    """
    arrival_rates = [4.]
    file_configs = [
        set(np.linspace(1, K, K, dtype=int)) for K in [200, 500]
    ]
    policies = ["SB-MS", "SB-RFwPMS"]

    for policy in policies:
        for arrival_rate in arrival_rates:
            for W in file_configs:
                master_file = W
                swarms_config = {
                    '1': {
                        'swarm_key': '1',
                        'arrival_rate': arrival_rate,
                        'primary_file': W,
                        'inter_swarm_behavior': 'altruistic',
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W)
                    }
                }
                swarms_config = set_inter_swarm_behavior(swarms_config=swarms_config, master_file=master_file)
                args = get_parser().parse_args(
                    [
                        "--results_directory", os.path.join(
                        "./../results/sojourn_times_single_swarm/", policy, "K_" + str(len(W))
                    ),
                        "--U", '1.',
                        "--mu", '1.',
                        "--mu_hat", '1.',
                        "--L", '1',
                        "--p", '0.',
                        "--initial_condition", "zero",
                        "--policy", policy,
                        "--end_time", '5000.',
                        "--seed", '1',
                    ]
                )
                args.master_file = master_file
                args.swarms_config = swarms_config
                args.is_autonomous = False
                args.Y_opt = True
                args.is_log_freq = True if len(W) <= 100 else False
                main(args)
    return


# %-----------------------------------------------------------------------------------%
# %----------------Flash-crowd Response-----------------------%
# %-----------------------------------------------------------------------------------%
def flash_crowd_response():
    """
    purpose: evaluate flash-crowd responses of SB-RFwPMS and SB-MS.
    @return:
    """
    policies = ["SB-MS", "SB-TMS", "SB-RNwPMS", "SB-RFwPMS"]
    arrival_rates = [0.]
    file_configs = [
        set(np.linspace(1, K, K, dtype=int)) for K in [100]
    ]
    for policy in policies:
        for arrival_rate in arrival_rates:
            for W in file_configs:
                master_file = W
                swarms_config = {
                    '1': {
                        'swarm_key': '1',
                        'arrival_rate': arrival_rate,
                        'primary_file': W,
                        'inter_swarm_behavior': 'altruistic',
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W)
                    }
                }
                swarms_config = set_inter_swarm_behavior(swarms_config=swarms_config, master_file=master_file)
                args = get_parser().parse_args(
                    [
                        "--results_directory", os.path.join(
                        "./../results/flash_crowd/", policy, "K_" + str(len(W))
                    ),
                        "--U", '1.',
                        "--mu", '1.',
                        "--mu_hat", '1.',
                        "--L", '1',
                        "--p", '0.',
                        "--initial_condition", "flash_crowd",
                        "--policy", policy,
                        "--end_time", '600.',
                        "--seed", '1',
                    ]
                )
                args.master_file = master_file
                args.swarms_config = swarms_config
                args.is_autonomous = False
                args.Y_opt = True
                args.is_log_freq = True
                main(args)
    return


# %-----------------------------------------------------------------------------------%
# %----------------One-club Escape-----------------------%
# %-----------------------------------------------------------------------------------%
def one_club_escape():
    """
    purpose: evaluate one-club escape characteristics of SB-RFwPMS and SB-MS.
    @return:
    """
    policies = ["SB-MS", "SB-RFwPMS"]
    arrival_rates = [20.]
    file_configs = [
        set(np.linspace(1, K, K, dtype=int)) for K in [10]
    ]

    for policy in policies:
        for arrival_rate in arrival_rates:
            for W in file_configs:
                master_file = W
                swarms_config = {
                    '1': {
                        'swarm_key': '1',
                        'arrival_rate': arrival_rate,
                        'primary_file': W,
                        'inter_swarm_behavior': 'altruistic',
                        'alpha': 1e-9,
                        'beta': 1.5,
                        'TMS_threshold': 2 * len(W)
                    }
                }
                swarms_config = set_inter_swarm_behavior(swarms_config=swarms_config, master_file=master_file)
                args = get_parser().parse_args(
                    [
                        "--results_directory", os.path.join(
                        "./../results/one_club_escape/", policy, "K_" + str(len(W))
                    ),
                        "--U", '1.',
                        "--mu", '1.',
                        "--mu_hat", '1.',
                        "--L", '1',
                        "--p", '0.',
                        "--initial_condition", "one_club",
                        "--policy", policy,
                        "--end_time", '200.',
                        "--seed", '1',
                    ]
                )
                args.master_file = master_file
                args.swarms_config = swarms_config
                args.is_autonomous = False
                args.Y_opt = True
                args.is_log_freq = True
                main(args)
    return


# %-----------------------------------------------------------------------------------%
# %----------------Parser-----------------------%
# %-----------------------------------------------------------------------------------%
def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description='Simulate an unstructured P2P file-sharing network.', add_help=add_help)
    parser.add_argument("--results_directory", help="directory for results of the simulation.",
                        type=str,
                        default="./../results/"
                        )
    # required=True)
    parser.add_argument("--master_file", help="master-file.",
                        type=set,
                        default=set(np.linspace(1, 100, 100, dtype=int))
                        )
    # required=True)
    parser.add_argument("--swarms_config", help="swarm_container and their individual configurations.",
                        type=json.loads,
                        default=single_swarm
                        )
    # parser.add_argument("--initial_cache", help="Cache of arriving peers. "
    #                                             "Options include empty or one_piece_random.",
    #                     type=str,
    #                     default="empty"
    #                     )
    # # required=True)
    # parser.add_argument("--persistence", help="persistence behavior at download completion."
    #                                           "Options include no, one_piece.",
    #                     type=str,
    #                     default="no"
    #                     )
    # # required=True)
    parser.add_argument("--is_autonomous", help="Whether or not swarm_container live autonomously.",
                        type=bool,
                        default=False
                        )
    # required=True)
    parser.add_argument("--U", help="contact rate of seed.",
                        type=float,
                        default=1.
                        )
    # required=True)
    parser.add_argument("--mu", help="contact rate of a (normal) peer's tit-for-tat link.",
                        type=float,
                        default=1.
                        )
    # required=True)
    parser.add_argument("--mu_hat", help="contact rate of a (normal) peer's optimistic-unchoke link.",
                        type=float,
                        default=1.
                        )
    # required=True)
    parser.add_argument("--L", help="number of links with each peer (including seed).", type=int,
                        default=1
                        )
    # required=True)
    parser.add_argument("--p", help="probability of altruism in case of no benefit in a tit-for-tat exchange.",
                        type=float,
                        default=0.
                        )
    # required=True)
    parser.add_argument("--Y_opt", help="whether or not to use optimistic-unchoke.", type=bool,
                        default=True
                        )
    # required=True)
    parser.add_argument("--initial_condition", help="Options are flash_crowd, one_club, zero.", type=str,
                        default="flash_crowd"
                        )
    # required=True)
    parser.add_argument("--policy", help="piece-selection policy",
                        type=str,
                        # default="SB-RFwPMS"
                        # default="SB-TMS"
                        default="SB-RNwPMS"
                        )
    # required=True)
    parser.add_argument("--end_time", help="End time of simulation.", type=float,
                        default=600.
                        )
    # required=True)
    parser.add_argument("--seed", help="seed for reproducing results", type=str,
                        default="1"
                        )
    # required=True)
    parser.add_argument("--is_log_freq", help="whether or not to log frequencies.", type=bool,
                        default=True
                        )
    # required=True)
    return parser


# %-----------------------------------------------------------------------------------%
# %----------------Main Function-----------------------%
# %-----------------------------------------------------------------------------------%
def main(args):
    # context update
    context = Context()
    context.results_directory = args.results_directory
    context.master_file = args.master_file
    context.swarms_config = args.swarms_config
    context.is_autonomous = args.is_autonomous
    context.U = args.U
    context.mu = args.mu
    context.mu_hat = args.mu_hat
    context.L = args.L
    context.p = args.p
    context.Y_opt = args.Y_opt
    context.initial_condition = args.initial_condition
    context.policy = args.policy
    context.end_time = args.end_time
    context.seed = int(args.seed)
    context.is_log_freq = args.is_log_freq

    # set all seeds
    same_seeds(context.seed)

    # create a subdirectory in results_directory for outputs of this run
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M")
    outputs_directory = os.path.join(
        args.results_directory, timestamp + '/')
    context.outputs_directory = outputs_directory
    context.plots_directory = os.path.join(context.outputs_directory, 'plots/')
    context.histories_directory = os.path.join(context.outputs_directory, 'histories/')
    for directory in [context.outputs_directory, context.histories_directory, context.plots_directory]:
        if os.path.exists(directory):
            pass
        else:
            os.makedirs(directory)

    # save experiment details
    context.L_tit_for_tat = context.L - 1 if context.Y_opt else context.L
    context.L_opt_unchoke = context.L - context.L_tit_for_tat
    history = dict()
    history['master_file'] = np.asarray(list(context.master_file)).tolist()
    history['swarms_config'] = context.swarms_config
    history['is_autonomous'] = context.is_autonomous
    history['U'] = context.U
    history['mu'] = context.mu
    history['mu_hat'] = context.mu_hat
    history['L'] = context.L
    history['L_tit_for_tat'] = context.L_tit_for_tat
    history['L_opt_unchoke'] = context.L_opt_unchoke
    history['p'] = context.p
    history['Y_opt'] = context.Y_opt
    history['initial_condition'] = context.initial_condition
    history['policy'] = context.policy
    history['end_time'] = context.end_time
    history['seed'] = context.seed
    history['is_log_freq'] = context.is_log_freq

    def set_default(obj):
        if isinstance(obj, set):
            return np.asarray(list(obj)).tolist()
        # raise TypeError

    with open(os.path.join(context.outputs_directory, 'experiment_details.json'), "w") as output_file:
        json.dump(history, output_file, default=set_default)

    # %-----------------------------------------------------------------------------------%
    # %-----------------------------------------------------------------------------------%
    # instantiate the network with specified initial_condition
    swarm_container = SwarmContainer(swarms_config=context.swarms_config)
    network = Network(
        context=context,
        master_file=context.master_file,
        swarm_container=swarm_container,
    )
    network.set_initial_condition(initial_condition=context.initial_condition)
    seed = Seed(network=network)
    from Logger import Logger
    logger = Logger(network=network, context=context)

    start = time.time()
    context.time = 0
    context.successful_peers = {key: [] for key in network.swarm_container.keys()}
    # log initial values for all histories
    logger.log_all_histories(network=network, context=context)

    with tqdm(total=context.end_time) as pbar:
        while context.time <= context.end_time:
            next_event = sample_next_event(network=network, context=context)
            context.time += next_event[-1]  # next_event[-1] = time_step

            # empty peer arrival
            if next_event[0] == "peer_arrival":
                swarm_key = next_event[1]
                network.append(
                    Peer(swarm=network.swarm_container[swarm_key], cache=set(), entry_time=context.time)
                )
                network.empty_peers_count += 1
                network.total_population += 1
                network.swarm_container[swarm_key].empty_peers_count += 1
                network.swarm_container[swarm_key].swarm_population += 1
                logger.log_event(event=('peer_arrival', swarm_key), network=network, context=context)
            # optimistic unchoke by seed
            elif next_event[0] == "seed_contact":
                seed.optimistic_unchoke(network=network, context=context, logger=logger)
            # a normal peer contacts another normal peer
            else:
                peer_index = next_event[1]
                if next_event[2] == "opt_unchoke":
                    network[peer_index].optimistic_unchoke(network=network, context=context, logger=logger)
                else:
                    network[peer_index].tit_for_tat_exchange(network=network, context=context, logger=logger)
            pbar.update(next_event[-1])

    # save the histories and plots
    from log_utils import save_histories, plot_and_save_histories
    save_histories(
        histories_directory=context.histories_directory,
        network_population=logger.network_population,
        network_empty_peers_count=logger.network_empty_peers_count,
        swarm_populations=logger.swarm_population,
        swarm_empty_peers_count=logger.swarm_empty_peers_count,
        network_chunk_counts=logger.network_chunk_counts,
        swarm_chunk_counts=logger.swarm_chunk_counts
    )

    sojourn_time = {}
    network_successful_peers = []
    for swarm_key in context.successful_peers.keys():
        sojourn_time[swarm_key] = {}
        # mean
        sojourn_time[swarm_key]['mean'] = np.mean([peer.get_sjn_time() for peer in context.successful_peers[swarm_key]])
        # standard deviation
        sojourn_time[swarm_key]['std'] = np.std([peer.get_sjn_time() for peer in context.successful_peers[swarm_key]])
        # confidence interval
        sojourn_time[swarm_key]['confidence_interval'] = [
            sojourn_time[swarm_key]['mean'] - 3.291 * (sojourn_time[swarm_key]['std']) / len(
                context.successful_peers[swarm_key]
            ),
            sojourn_time[swarm_key]['mean'] + 3.291 * (sojourn_time[swarm_key]['std']) / len(
                context.successful_peers[swarm_key]
            )
        ]
        network_successful_peers.extend(context.successful_peers[swarm_key])
        # total_sum += np.sum([peer.get_sjn_time() for peer in context.successful_peers[swarm_key]])
        # total_count += len(context.successful_peers[swarm_key])
    sojourn_time['network'] = {}
    # mean for network
    sojourn_time['network']['mean'] = np.mean([peer.get_sjn_time() for peer in network_successful_peers])
    # standard deviation for network
    sojourn_time['network']['std'] = np.std([peer.get_sjn_time() for peer in network_successful_peers])
    # confidence interval for network
    sojourn_time['network']['confidence_interval'] = [
        sojourn_time['network']['mean'] - 3.291 * (sojourn_time['network']['std']) / len(network_successful_peers),
        sojourn_time['network']['mean'] + 3.291 * (sojourn_time['network']['std']) / len(network_successful_peers)
    ]
    # log the total time of the simulation
    sojourn_time['simulation_time_min'] = (time.time() - start) / 60.
    with open(os.path.join(context.outputs_directory, 'sojourn_times.json'), "w") as output_file:
        json.dump(sojourn_time, output_file)

    plot_and_save_histories(
        plots_directory=context.plots_directory, network_population=logger.network_population,
        network_empty_peers_count=logger.network_empty_peers_count,
        swarm_population=logger.swarm_population,
        swarm_empty_peers_count=logger.swarm_empty_peers_count,
        network_chunk_counts=logger.network_chunk_counts,
        swarm_chunk_counts=logger.swarm_chunk_counts,
        show_plots=False
    )

    return


# %-----------------------------------------------------------------------------------%
# %----------------Call to Main Function-----------------------%
# %-----------------------------------------------------------------------------------%
if __name__ == "__main__":
    # args = get_parser().parse_args()

    # Stability Check
    print("---Starting Stability Check---")
    stability_check()
    print("---Stability Check Complete---")

    # # Scalability Check
    # print("---Starting Scalability Check---")
    # scalability_check()
    # print("---Scalability Check Complete---")
    #
    # # Sojourn Time Performance
    # print("---Starting Sojourn Time Performance Check---")
    # sojourn_time_performance()
    # print("---Sojourn Time Performance Check Complete---")

    # Single Swarm Expected Steady State Sojourn Times
    # print("---Starting Single Swarm Sojourn Times Check---")
    # single_swarm_sojourn_times()
    # print("---Single Swarm Sojourn Times Check Complete---")

    # Single Swarm Flash-crowd Response
    # print("---Starting Flash-Crowd Response Check---")
    # flash_crowd_response()
    # print("---Flash-crowd Response Check Complete---")

    # Single Swarm One-club Escape
    # print("---Starting One-club Escape Check---")
    # one_club_escape()
    # print("---One-club Escape Check Complete---")
