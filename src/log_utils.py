import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


# %-----------------------------------------------------------------------------------%
# %----------------Saving Functions-----------------------%
# %-----------------------------------------------------------------------------------%
def save_network_population(output_directory: str, network_population: dict):
    """
    purpose: save network population history in csv file.
    @param output_directory:
    @param network_population:
    @return:
    """
    # save to csv file
    time = network_population['time']
    pop = network_population['population']
    df = pd.DataFrame([time, pop], index=['time', 'population']).T
    df.to_csv(
        os.path.join(output_directory, '1_network_population.csv')
    )
    return


def save_network_empty_peers_count(output_directory: str, network_empty_peers_count: dict):
    """
    purpose: save network's empty_peers_count history in csv file.
    @param output_directory:
    @param network_empty_peers_count:
    @return:
    """
    # save to csv file
    time = network_empty_peers_count['time']
    empty_peers_count = network_empty_peers_count['empty_peers_count']
    pop = network_empty_peers_count['population']
    df = pd.DataFrame([time, empty_peers_count, pop],
                      index=['time', 'empty_peers_count', 'population']).T
    df.to_csv(
        os.path.join(output_directory, '2_network_empty_peers_count.csv')
    )
    return


def save_swarm_populations(output_directory: str, swarm_population: dict):
    """
    purpose: save swarm population histories in csv files.
    @param output_directory:
    @param swarm_population:
    @return:
    """
    # save to csv file
    df_dict = {}
    df = pd.DataFrame()
    for key in swarm_population.keys():
        time = swarm_population[key]['time']
        population = swarm_population[key]['population']
        df_dict[key] = pd.DataFrame([time, population],
                                    index=['time (swarm-' + key + ')', 'population (swarm-' + key + ')']).T
        df = pd.concat([df, df_dict[key]])
    df.to_csv(
        os.path.join(output_directory, '3_swarm_populations.csv')
    )
    return


def save_swarm_empty_peers_count(output_directory: str, swarm_empty_peers_count: dict):
    """
    purpose: save swarm empty_peers_count histories in csv file.
    @param output_directory:
    @param swarm_empty_peers_count:
    @return:
    """
    # save to csv file
    df_dict = {}
    df = pd.DataFrame()
    for key in swarm_empty_peers_count.keys():
        time = swarm_empty_peers_count[key]['time']
        empty_peers_count = swarm_empty_peers_count[key]['empty_peers_count']
        pop = swarm_empty_peers_count[key]['population']
        df_dict[key] = pd.DataFrame([time, empty_peers_count, pop],
                                    index=['time (swarm-' + key + ')',
                                           'empty_peers_count (swarm-' + key + ')',
                                           'population (swarm-' + key + ')'
                                           ]).T
        df = pd.concat([df, df_dict[key]])
    df.to_csv(
        os.path.join(output_directory, '4_swarm_empty_peers_counts.csv')
    )
    return


def save_network_chunk_counts(output_directory: str, network_chunk_counts: dict):
    """
    purpose: save network chunk counts in csv file.
    @param output_directory:
    @param network_chunk_counts:
    @return:
    """
    # save only mode_count and rarest_count to csv file
    time = network_chunk_counts['time']
    pop = np.asarray(network_chunk_counts['population'], dtype=np.float64)
    to_be_array = []
    for c in network_chunk_counts.keys():
        if c == 'time' or c == 'population':
            continue
        to_be_array.append(network_chunk_counts[c])
    chunk_counts_array = np.asarray(to_be_array, dtype=np.float64)
    mode_counts = np.max(chunk_counts_array, axis=0)
    rarest_counts = np.min(chunk_counts_array, axis=0)
    try:
        mode_freq = np.divide(mode_counts, pop, out=np.zeros_like(pop), where=pop != 0)
        rarest_freq = np.divide(rarest_counts, pop, out=np.zeros_like(pop), where=pop != 0)
    except ValueError:
        mode_freq = np.asarray([])
        rarest_freq = np.asarray([])
    except:
        mode_freq = np.asarray([])
        rarest_freq = np.asarray([])

    df = pd.DataFrame([time, pop, mode_counts, rarest_counts, mode_freq, rarest_freq],
                      index=['time', 'population', 'mode_count', 'rarest_count', 'mode_freq', 'rarest_freq']).T
    df.to_csv(
        os.path.join(output_directory, '5_network_chunk_counts.csv')
    )


def save_swarm_chunk_counts(output_directory: str, swarm_chunk_counts: dict):
    """
    purpose: save swarm chunk counts in csv files.
    @param output_directory:
    @param swarm_chunk_counts:
    @return:
    """
    for key in swarm_chunk_counts.keys():
        time = swarm_chunk_counts[key]['time']
        pop = np.asarray(swarm_chunk_counts[key]['population'], dtype=np.float64)
        to_be_array = []
        for c in swarm_chunk_counts[key]['primary_file']:
            to_be_array.append(swarm_chunk_counts[key][c])
        chunk_counts_array = np.asarray(to_be_array, dtype=np.float64)
        mode_counts = np.max(chunk_counts_array, axis=0)
        rarest_counts = np.min(chunk_counts_array, axis=0)
        try:
            mode_freq = np.divide(mode_counts, pop, out=np.zeros_like(pop), where=pop != 0)
            rarest_freq = np.divide(rarest_counts, pop, out=np.zeros_like(pop), where=pop != 0)
        except ValueError:
            mode_freq = np.asarray([])
            rarest_freq = np.asarray([])
        except:
            mode_freq = np.asarray([])
            rarest_freq = np.asarray([])

        df = pd.DataFrame([time, pop, mode_counts, rarest_counts, mode_freq, rarest_freq],
                          index=['time', 'population', 'mode_count', 'rarest_count', 'mode_freq', 'rarest_freq']).T
        df.to_csv(
            os.path.join(output_directory, '6_swarm_' + key + '_chunk_counts.csv')
        )


def save_histories(histories_directory: str,
                   network_population: dict,
                   network_empty_peers_count: dict,
                   swarm_populations: dict,
                   swarm_empty_peers_count: dict,
                   network_chunk_counts: dict = None,
                   swarm_chunk_counts: dict = None
                   ):
    """
    :purpose: save all histories.
    :return:
    """
    save_network_population(histories_directory, network_population)
    save_network_empty_peers_count(histories_directory, network_empty_peers_count)
    save_swarm_populations(histories_directory, swarm_populations)
    save_swarm_empty_peers_count(histories_directory, swarm_empty_peers_count)
    if network_chunk_counts is not None:
        save_network_chunk_counts(histories_directory, network_chunk_counts)
    if swarm_chunk_counts is not None:
        save_swarm_chunk_counts(histories_directory, swarm_chunk_counts)
    return


# %-----------------------------------------------------------------------------------%
# %----------------Plot FUNCTIONS-----------------------%
# %-----------------------------------------------------------------------------------%
def plot_and_save_network_population(plot_directory: str, network_population: dict, show_plots: bool = True):
    # plot network population
    plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), dpi=300)
    plt.plot(network_population['time'], network_population['population'], linewidth=1.5, color='b')
    # plt.title("Network Population")
    plt.xlabel("time")
    plt.ylabel("network population")
    plt.grid()
    plt.savefig(os.path.join(plot_directory, '1_network_population.png'), format='png', dpi=300)
    if show_plots:
        plt.show()
    plt.close('all')
    return


def plot_and_save_network_empty_peers_count(plot_directory: str, network_empty_peers_count: dict,
                                            show_plots: bool = True):
    # plot empty_peers_count
    fig, ax1 = plt.subplots(figsize=(1.5 * 6.4, 1.5 * 4.8), dpi=300)
    ax1.grid()
    ax2 = ax1.twinx()
    time = network_empty_peers_count['time']
    empty_peers_count = network_empty_peers_count['empty_peers_count']
    pop = np.asarray(network_empty_peers_count['population'], dtype=np.float64)
    l1, = ax1.plot(
        time,
        pop,
        linewidth=2,
        linestyle='-',
        color='b',
        label="Network Population"
    )
    ratio_of_empty_peers = np.divide(empty_peers_count, pop, out=np.zeros_like(pop), where=pop != 0)
    l2, = ax2.plot(
        time,
        ratio_of_empty_peers,
        linewidth=2,
        linestyle='-.',
        color='r',
        label="Ratio of Empty Peers"
    )
    ax1.set_xlabel('time')
    ax1.set_ylabel('network population')
    ax2.set_ylabel('ratio of empty-peers')
    plt.legend([l1, l2], ['network population', 'ratio of empty-peers'], loc="lower right")
    plt.savefig(os.path.join(plot_directory, '2_network_empty_peers_count.png'), format='png', dpi=300)
    if show_plots:
        plt.show()
    plt.close('all')
    return


def plot_and_save_swarm_populations(plot_directory: str, swarm_population: dict, show_plots: bool = True):
    # plot swarm populations
    plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), dpi=300)
    for key in swarm_population.keys():
        plt.plot(
            swarm_population[key]['time'], swarm_population[key]['population'],
            linewidth=1.5,
            label='swarm-' + key
        )
    # plt.title("Swarm Populations")
    plt.xlabel("time")
    plt.ylabel("swarm population")
    plt.grid()
    plt.legend(loc="upper right", prop={'size': 10})
    plt.savefig(os.path.join(plot_directory, '3_swarm_populations.png'))
    if show_plots:
        plt.show()
    plt.close('all')
    return


def plot_and_save_network_chunk_counts(plot_directory: str, network_chunk_counts: dict, show_plots: bool = True):
    time = network_chunk_counts['time']
    pop = np.asarray(network_chunk_counts['population'], dtype=np.float64)
    to_be_array = []
    for c in network_chunk_counts.keys():
        if c == 'time' or c == 'population':
            continue
        to_be_array.append(network_chunk_counts[c])
    chunk_counts_array = np.asarray(to_be_array, dtype=np.float64)
    mode_counts = np.max(chunk_counts_array, axis=0)
    rarest_counts = np.min(chunk_counts_array, axis=0)
    try:
        mode_freq = np.divide(mode_counts, pop, out=np.zeros_like(pop), where=pop != 0)
        rarest_freq = np.divide(rarest_counts, pop, out=np.zeros_like(pop), where=pop != 0)
    except ValueError:
        mode_freq = np.asarray([])
        rarest_freq = np.asarray([])
    except:
        mode_freq = np.asarray([])
        rarest_freq = np.asarray([])

    plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), dpi=300)
    plt.plot(
        # time, mode_freq,
        time, mode_counts,
        linewidth=1.5,
        linestyle='-',
        # label='mode-freq',
        label='mode-count',
        color='r'
    )
    plt.plot(
        # time, rarest_freq,
        time, rarest_counts,
        linewidth=1.5,
        linestyle='-.',
        # label='rarest-chunk freq',
        label='rarest chunk-count',
        color='g'
    )
    # plt.title("Network Chunk Frequencies")
    plt.xlabel("time")
    plt.ylabel("network chunk-counts")
    plt.grid()
    plt.legend(loc="upper right", prop={'size': 10})
    plt.savefig(os.path.join(plot_directory, '5_network_chunk_counts.png'))
    if show_plots:
        plt.show()
    plt.close('all')

    plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), dpi=300)
    plt.plot(
        time, mode_freq,
        # time, mode_counts,
        linewidth=1.5,
        linestyle='-',
        label='mode-freq',
        # label='mode-count',
        color='r'
    )
    plt.plot(
        time, rarest_freq,
        # time, rarest_counts,
        linewidth=1.5,
        linestyle='-.',
        label='rarest-chunk freq',
        # label='rarest chunk-count',
        color='g'
    )
    # plt.title("Network Chunk Frequencies")
    plt.xlabel("time")
    plt.ylabel("network chunk-freqs")
    plt.grid()
    plt.legend(loc="upper right", prop={'size': 10})
    plt.savefig(os.path.join(plot_directory, '5_network_chunk_freqs.png'))
    if show_plots:
        plt.show()
    plt.close('all')
    return


def plot_and_save_swarm_chunk_counts(plot_directory: str, swarm_chunk_counts: dict, show_plots: bool = True):
    for key in swarm_chunk_counts.keys():
        time = swarm_chunk_counts[key]['time']
        pop = np.asarray(swarm_chunk_counts[key]['population'], dtype=np.float64)
        to_be_array = []
        for c in swarm_chunk_counts[key]['primary_file']:
            to_be_array.append(swarm_chunk_counts[key][c])
        chunk_counts_array = np.asarray(to_be_array, dtype=np.float64)
        mode_counts = np.max(chunk_counts_array, axis=0)
        rarest_counts = np.min(chunk_counts_array, axis=0)
        try:
            mode_freq = np.divide(mode_counts, pop, out=np.zeros_like(pop), where=pop != 0)
            rarest_freq = np.divide(rarest_counts, pop, out=np.zeros_like(pop), where=pop != 0)
        except ValueError:
            mode_freq = np.asarray([])
            rarest_freq = np.asarray([])
        except:
            mode_freq = np.asarray([])
            rarest_freq = np.asarray([])

        plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), dpi=300)
        plt.plot(
            # time, mode_freq,
            time, mode_counts,
            linewidth=1.5,
            linestyle='-',
            # label='mode-freq',
            label='mode-count',
            color='r'
        )
        plt.plot(
            # time, rarest_freq,
            time, rarest_counts,
            linewidth=1.5,
            linestyle='-.',
            # label='rarest-chunk freq',
            label='rarest chunk-count',
            color='g'
        )
        # plt.title("Swarm-"+key+" Chunk Frequencies")
        plt.xlabel("time")
        plt.ylabel("swarm-" + key + " chunk-counts")
        plt.grid()
        plt.legend(loc="upper right", prop={'size': 10})
        plt.savefig(os.path.join(plot_directory, '6_swarm_' + key + '_chunk_counts.png'))
        if show_plots:
            plt.show()

        plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), dpi=300)
        plt.plot(
            time, mode_freq,
            # time, mode_counts,
            linewidth=1.5,
            linestyle='-',
            label='mode-freq',
            # label='mode-count',
            color='r'
        )
        plt.plot(
            time, rarest_freq,
            # time, rarest_counts,
            linewidth=1.5,
            linestyle='-.',
            label='rarest-chunk freq',
            # label='rarest chunk-count',
            color='g'
        )
        # plt.title("Swarm-"+key+" Chunk Frequencies")
        plt.xlabel("time")
        plt.ylabel("swarm-" + key + " chunk-freqs")
        plt.grid()
        plt.legend(loc="upper right", prop={'size': 10})
        plt.savefig(os.path.join(plot_directory, '6_swarm_' + key + '_chunk_freqs.png'))
        if show_plots:
            plt.show()
        plt.close('all')


def plot_and_save_histories(plots_directory: str,
                            network_population: dict,
                            network_empty_peers_count: dict,
                            swarm_population: dict,
                            swarm_empty_peers_count: dict,
                            network_chunk_counts: dict = None,
                            swarm_chunk_counts: dict = None,
                            show_plots=True
                            ):
    """
    :purpose: plot all histories and save figures.
    :return:
    """
    plot_and_save_network_population(plots_directory, network_population, show_plots=show_plots)
    plot_and_save_network_empty_peers_count(
        plots_directory,
        network_empty_peers_count,
        show_plots=show_plots
    )
    plot_and_save_swarm_populations(plots_directory, swarm_population, show_plots=show_plots)
    if network_chunk_counts is not None:
        plot_and_save_network_chunk_counts(plots_directory, network_chunk_counts, show_plots=show_plots)
    if swarm_chunk_counts is not None:
        plot_and_save_swarm_chunk_counts(plots_directory, swarm_chunk_counts, show_plots=show_plots)
    return
