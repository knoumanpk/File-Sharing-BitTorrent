import numpy as np
import math


# %-----------------------------------------------------------------------------------%
# %----------------Implementation of Necessary Functions-----------------------%
# %-----------------------------------------------------------------------------------%
def exchange_chunks(peer: dict, network, context, logger):
    """
    purpose: execute chunk-exchange between two peers.
    @param peer: A dictionary with keys 1 and 2. Values are the two corresponding peers.
    @param network: Network to which peers belong.
    @param context: Context to pass across functions.
    @param logger: Logger to log histories.
    @return:
    """
    W, S, ally_set, swarm_key = {}, {}, {}, {}
    for idx in peer.keys():  # idx = 1 or 2
        W[idx] = peer[idx].primary_file
        S[idx] = peer[idx].cache
        ally_set[idx] = peer[idx].ally_set
        swarm_key[idx] = peer[idx].swarm_key

    S_hat = {}  # revealed cache profiles
    for i in range(1, 3):  # i=1, 2; j=-i
        j = 2 if i == 1 else 1
        if swarm_key[j] not in ally_set[i]:
            S_hat[i] = set()
        else:
            S_hat[i] = S[i]

    A_trans = {}  # transferable set
    for i in range(1, 3):
        j = 2 if i == 1 else 1
        if S_hat[j].intersection(W[i]).difference(S[i]):
            A_trans[j] = compute_A_trans(
                revealed_cache=S_hat[i],
                contacted_peer=peer[j],
                network=network,
                context=context
            )
        else:
            if np.random.binomial(n=1, p=context.p):
                A_trans[j] = compute_A_trans(
                    revealed_cache=S_hat[i],
                    contacted_peer=peer[j],
                    network=network,
                    context=context
                )
            else:
                A_trans[j] = set()

    download_from_A_trans(
        peer=peer,
        A_trans=A_trans,
        network=network,
        context=context,
        logger=logger
    )
    return


def download_from_A_trans(peer: dict, A_trans: dict, network, context, logger):
    """
    purpose: execute the chunk-exchange between two peers.
    @param peer: A dictionary with keys 1 and 2. Values are the two corresponding peers.
    @param A_trans: A dictionary with keys 1 and 2. Values are the corresponding transfer-sets.
    @param network: Network to which peers belong.
    @param context: Context to pass around functions.
    @param logger: Logger to log histories.
    @return:
    """
    # if both A_trans sets are empty, do nothing and return
    if not any(bool(s) for s in A_trans.values()):
        return
    # otherwise, at least one chunk-download happens
    event_name = "chunk_download"
    event_desc = {}
    for idx in peer.keys():
        # Continue to next idx if A_trans[idx] is empty
        if not A_trans[idx]:
            continue
        # store swarm_key to remember which swarm's values to log
        event_desc[idx] = peer[idx].swarm_key
        # add the piece to peer's cache
        peer[idx].cache = peer[idx].cache.union(A_trans[idx])
        # check whether peer stays or departs
        W = peer[idx].primary_file
        S = peer[idx].cache
        if W.difference(S):  # peer stays
            # update chunk_counts in respective swarm and network
            for c in A_trans[idx]:
                network.swarm_container[peer[idx].swarm_key].chunk_counts[c] += 1
                network.chunk_counts[c] += 1
            # network and swarm population remain same
        else:  # peer departs
            # update chunk_counts in respective swarm and the network
            for c in peer[idx].cache.difference(A_trans[idx]):
                network.swarm_container[peer[idx].swarm_key].chunk_counts[c] -= 1
                network.chunk_counts[c] -= 1
            # update network population and swarm population
            network.total_population -= 1
            network.swarm_container[peer[idx].swarm_key].swarm_population -= 1
            # remove the peer from the network and append it to the list of successful peers
            network.remove(peer[idx])
            context.successful_peers[peer[idx].swarm_key].append(peer[idx])
            # log the exit time of the peer
            peer[idx].exit_time = context.time
        # log empty peers if peer's cache was empty before the download
        if len(peer[idx].cache) == 1:
            network.empty_peers_count -= 1
            network.swarm_container[peer[idx].swarm_key].empty_peers_count -= 1
    # log histories
    logger.log_event(event=(event_name, event_desc), network=network, context=context)
    return


def compute_A_trans(revealed_cache: set, contacted_peer, network, context):
    """
    purpose: compute the transferable set for contacted peer from the revealed cache.
    @param revealed_cache: cache revealed to the contacted peer.
    @param contacted_peer: peer which will download a chunk from revealed_cache.
    @param network: Network to which contacted_peer belongs.
    @param context: Context to carry across functions.
    @return: A_trans
    """
    if context.policy == "SB-RFwPMS":
        return compute_A_trans_via_SB_RFwPMS(
            revealed_cache=revealed_cache,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )
    elif context.policy == "SB-RNwPMS":
        return compute_A_trans_via_SB_RNwPMS(
            revealed_cache=revealed_cache,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )
    elif context.policy == "SB-TMS":
        return compute_A_trans_via_SB_TMS(
            revealed_cache=revealed_cache,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )
    elif context.policy == "SB-MS":
        return compute_A_trans_via_SB_MS(
            revealed_cache=revealed_cache,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )
    elif context.policy == "SB-RF":
        return compute_A_trans_via_SB_RN(
            revealed_cache=revealed_cache,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )
    else:
        return compute_A_trans_via_SB_RN(
            revealed_cache=revealed_cache,
            contacted_peer=contacted_peer,
            network=network,
            context=context
        )


def compute_A_trans_via_SB_RFwPMS(revealed_cache: set, contacted_peer, network, context):
    """
    purpose: compute the transferable set for contacted peer from the revealed cache using SB-RFwPMS.
    @param revealed_cache: cache revealed to the contacted peer.
    @param contacted_peer: peer which will download a chunk from revealed_cache.
    @param network: Network to which contacted_peer belongs.
    @param context: Context to carry across functions.
    @return: A_trans
    """
    T = revealed_cache
    W = contacted_peer.primary_file
    S = contacted_peer.cache
    F_W = contacted_peer.file
    R_W = contacted_peer.swarm.get_set_of_rare_pieces()
    H_1 = T.intersection(R_W).difference(S)
    H_2 = T.intersection(W).difference(S)
    H_3 = T.intersection(F_W).difference(S.union(W))
    if H_1:
        H = contacted_peer.swarm.get_rarest_pieces(H_1)
        A_trans = {np.random.choice([c for c in H])}
    elif H_2:
        chosen_chunk = np.random.choice([c for c in H_2])
        mode_count = contacted_peer.swarm.get_mode_count()
        rarest_count = contacted_peer.swarm.get_rarest_count()
        # largest mismatch
        largest_mismatch = mode_count - rarest_count
        # complementary chunk-count
        comp_chunk_count = 0
        for ally_swarm_key in contacted_peer.swarm.ally_set.difference({contacted_peer.swarm_key}):
            if chosen_chunk in network.swarm_container[ally_swarm_key].chunk_counts.keys():
                comp_chunk_count += network.swarm_container[ally_swarm_key].chunk_counts[chosen_chunk]
            else:
                pass
        alpha = contacted_peer.swarm.alpha
        beta = contacted_peer.swarm.beta
        K_W = len(W)
        zeta = math.exp(
            (-1 / beta) * ((largest_mismatch + comp_chunk_count ** alpha) / K_W)
        )
        if np.random.binomial(1, zeta):
            A_trans = {chosen_chunk}
        else:
            A_trans = {np.random.choice([c for c in H_3])} if H_3 else set()
    else:
        A_trans = {np.random.choice([c for c in H_3])} if H_3 else set()
    return A_trans


def compute_A_trans_via_SB_RNwPMS(revealed_cache: set, contacted_peer, network, context):
    """
    purpose: compute the transferable set for contacted peer from the revealed cache using SB-RNwPMS.
    @param revealed_cache: cache revealed to the contacted peer.
    @param contacted_peer: peer which will download a chunk from revealed_cache.
    @param network: Network to which contacted_peer belongs.
    @param context: Context to carry across functions.
    @return: A_trans
    """
    T = revealed_cache
    W = contacted_peer.primary_file
    S = contacted_peer.cache
    F_W = contacted_peer.file
    R_W = contacted_peer.swarm.get_set_of_rare_pieces()
    H_1 = T.intersection(R_W).difference(S)
    H_2 = T.intersection(W).difference(S)
    H_3 = T.intersection(F_W).difference(S.union(W))
    if H_1:
        A_trans = {np.random.choice([c for c in H_1])}
    elif H_2:
        chosen_chunk = np.random.choice([c for c in H_2])
        mode_count = contacted_peer.swarm.get_mode_count()
        rarest_count = contacted_peer.swarm.get_rarest_count()
        # largest mismatch
        largest_mismatch = mode_count - rarest_count
        # complementary chunk-count
        comp_chunk_count = 0
        for ally_swarm_key in contacted_peer.swarm.ally_set.difference({contacted_peer.swarm_key}):
            if network.swarm_container[ally_swarm_key].chunk_counts.has_key(chosen_chunk):
                comp_chunk_count += network.swarm_container[ally_swarm_key].chunk_counts[chosen_chunk]
            else:
                pass
        alpha = contacted_peer.swarm.alpha
        beta = contacted_peer.swarm.beta
        K_W = len(W)
        zeta = math.exp(
            (-1 / beta) * ((largest_mismatch + comp_chunk_count ** alpha) / K_W)
        )
        if np.random.binomial(1, zeta):
            A_trans = {chosen_chunk}
        else:
            A_trans = {np.random.choice([c for c in H_3])} if H_3 else set()
    else:
        A_trans = {np.random.choice([c for c in H_3])} if H_3 else set()
    return A_trans


def compute_A_trans_via_SB_TMS(revealed_cache: set, contacted_peer, network, context):
    """
    purpose: compute the transferable set for contacted peer from the revealed cache using SB-TMS.
    @param revealed_cache: cache revealed to the contacted peer.
    @param contacted_peer: peer which will download a chunk from revealed_cache.
    @param network: Network to which contacted_peer belongs.
    @param context: Context to carry across functions.
    @return: A_trans
    """
    T = revealed_cache
    W = contacted_peer.primary_file
    S = contacted_peer.cache
    F_W = contacted_peer.file
    R_W = contacted_peer.swarm.get_set_of_rare_pieces()
    H_1 = T.intersection(R_W).difference(S)
    H_2 = T.intersection(W).difference(S)
    H_3 = T.intersection(F_W).difference(S.union(W))
    if H_1:
        A_trans = {np.random.choice([c for c in H_1])}
    elif H_2:
        chosen_chunk = np.random.choice([c for c in H_2])
        mode_count = contacted_peer.swarm.get_mode_count()
        rarest_count = contacted_peer.swarm.get_rarest_count()
        # largest mismatch
        largest_mismatch = mode_count - rarest_count
        if largest_mismatch <= contacted_peer.swarm.TMS_threshold:
            A_trans = {chosen_chunk}
        else:
            A_trans = {np.random.choice([c for c in H_3])} if H_3 else set()
    else:
        A_trans = {np.random.choice([c for c in H_3])} if H_3 else set()
    return A_trans


def compute_A_trans_via_SB_MS(revealed_cache: set, contacted_peer, network, context):
    """
    purpose: compute the transferable set for contacted peer from the revealed cache using SB-TMS.
    @param revealed_cache: cache revealed to the contacted peer.
    @param contacted_peer: peer which will download a chunk from revealed_cache.
    @param network: Network to which contacted_peer belongs.
    @param context: Context to carry across functions.
    @return: A_trans
    """
    T = revealed_cache
    W = contacted_peer.primary_file
    S = contacted_peer.cache
    F_W = contacted_peer.file
    R_W = contacted_peer.swarm.get_set_of_rare_pieces()
    H_1 = T.intersection(R_W).difference(S)
    A_trans = {np.random.choice([c for c in H_1])} if H_1 else set()
    return A_trans


def compute_A_trans_via_SB_RF(revealed_cache: set, contacted_peer, network, context):
    """
    purpose: compute the transferable set for contacted peer from the revealed cache using SB-RF.
    @param revealed_cache: cache revealed to the contacted peer.
    @param contacted_peer: peer which will download a chunk from revealed_cache.
    @param network: Network to which contacted_peer belongs.
    @param context: Context to carry across functions.
    @return: A_trans
    """
    T = revealed_cache
    W = contacted_peer.primary_file
    S = contacted_peer.cache
    F_W = contacted_peer.file
    H_1 = T.intersection(W).difference(S)
    H_2 = T.intersection(F_W).difference(S.union(W))
    if H_1:
        H = contacted_peer.swarm.get_rarest_pieces(H_1)
        A_trans = {np.random.choice([c for c in H])}
    else:
        A_trans = {np.random.choice([c for c in H_2])} if H_2 else set()
    return A_trans


def compute_A_trans_via_SB_RN(revealed_cache: set, contacted_peer, network, context):
    """
    purpose: compute the transferable set for contacted peer from the revealed cache using SB-RN.
    @param revealed_cache: cache revealed to the contacted peer.
    @param contacted_peer: peer which will download a chunk from revealed_cache.
    @param network: Network to which contacted_peer belongs.
    @param context: Context to carry across functions.
    @return: A_trans
    """
    T = revealed_cache
    W = contacted_peer.primary_file
    S = contacted_peer.cache
    F_W = contacted_peer.file
    H_1 = T.intersection(W).difference(S)
    H_2 = T.intersection(F_W).difference(S.union(W))
    if H_1:
        A_trans = {np.random.choice([c for c in H_1])}
    else:
        A_trans = {np.random.choice([c for c in H_2])} if H_2 else set()
    return A_trans
