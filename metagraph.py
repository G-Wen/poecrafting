import json
from collections import defaultdict
from functools import reduce
import networkx as nx
import numpy as np

def parse_mods(item_json):
    p_m2mg = {}
    s_m2mg = {}
    p_m_weights = {}
    s_m_weights = {}
    p_mg_weights = {}
    s_mg_weights = {}

    for mg, mods in item_json['modpool']['prefix'].items():
        for mod, tiers in mods.items():
            p_m2mg[mod] = mg
            p_m_weights[mod] = sum(tier['weight'] for tier in tiers)
        p_mg_weights[mg] = sum(p_m_weights[mod] for mod in mods)

    for mg, mods in item_json['modpool']['suffix'].items():
        for mod, tiers in mods.items():
            s_m2mg[mod] = mg
            s_m_weights[mod] = sum(tier['weight'] for tier in tiers)
        s_mg_weights[mg] = sum(s_m_weights[mod] for mod in mods)

    total_prefix_weight = sum(p_mg_weights.values())
    total_suffix_weight = sum(s_mg_weights.values())

    p_craftable = [k for k in item_json['craftable']['prefix']]
    s_craftable = [k for k in item_json['craftable']['suffix']]

    return p_m2mg, s_m2mg, p_m_weights, s_m_weights, p_mg_weights, s_mg_weights, total_prefix_weight, total_suffix_weight, p_craftable, s_craftable


def parse_required(item_json):
    p_m2mg, s_m2mg, p_m_weights, s_m_weights, p_mg_weights, s_mg_weights, total_prefix_weight, total_suffix_weight, p_craftable, s_craftable = parse_mods(item_json)
    requirements = item_json['required']
    mandatory = requirements['mandatory'][0] if requirements['mandatory'] else None
    req_affix_type = 'prefix' if requirements['prefix'] else 'suffix'
    req_affix_mg = p_m2mg if req_affix_type == 'prefix' else s_m2mg

    """
    We make the assumption that each item:
    1. Has at least two desired mods.
    2. There are exactly two mod groups.
    3. Has at least one mod in each group.
    """
    required_mods = []
    for first_m, first_ilvl in requirements[req_affix_type]['1'].items():
        required_mods.append(f"{req_affix_mg[first_m]},{first_m},{first_ilvl}")
    for second_m, second_ilvl in requirements[req_affix_type]['2'].items():
        required_mods.append(f"{req_affix_mg[second_m]},{second_m},{second_ilvl}")

    required_combos = []
    for first_m, first_ilvl in requirements[req_affix_type]['1'].items():
        for second_m, second_ilvl in requirements[req_affix_type]['2'].items():
            first_affix_string = f"{req_affix_mg[first_m]},{first_m},{first_ilvl}"
            second_affix_string = f"{req_affix_mg[second_m]},{second_m},{second_ilvl}"

            # Sort the first and second affix string alphabetically
            if first_affix_string > second_affix_string:
                first_affix_string, second_affix_string = second_affix_string, first_affix_string

            required_combos.append(f"{first_affix_string}|{second_affix_string}")
    """
    I'm currently unsure what the 'required' field is since a user can put a required mod as the only mod of a group.
    for required_group in requirements[req_affix_type]:
        req_strings = [f"{req_affix_mg[k]},{k},{v}" for k, v in requirements[req_affix_type][required_group].items()]
        for combo in itertools.combinations(req_strings, 1):
            if not mandatory or mandatory in [c.split(",")[0] for c in combo]:
                combos.append("|".join(combo))
    """
    return required_mods, required_combos


def get_blocking_affix(mg_weights, craftable_mgs):
    """
    :param mg_weights: mod group weights
    :param craftable_mgs: craftable mod groups
    :return: mod group with highest weight, weight
    """
    max_weight = -1
    block_mg = ''

    for mg in craftable_mgs:
        if mg not in mg_weights:
            # blocks no mods
            continue
        if mg_weights[mg] > max_weight:
            max_weight = mg_weights[mg]
            block_mg = mg

    return block_mg, max_weight


def generate_graph(item_json):
    p_m2mg, s_m2mg, p_m_weights, s_m_weights, p_mg_weights, s_mg_weights, total_prefix_weight, total_suffix_weight, p_craftable, s_craftable = parse_mods(item_json)
    required_mods, required_combos = parse_required(item_json)
    req_affix_type = 'prefix' if item_json['required']['prefix'] else 'suffix'
    if req_affix_type == 'prefix':
        mod_groups = item_json['modpool']['prefix']
        mg_weights = p_mg_weights
        total_weight = total_prefix_weight
        metamod_string = "ItemGenerationCannotChangeSuffixes,2779,1"
        metamod_mg, metamod_m, metamod_ilvl = metamod_string.split(",")
    else:
        mod_groups = item_json['modpool']['suffix']
        mg_weights = s_mg_weights
        total_weight = total_suffix_weight
        metamod_string = "ItemGenerationCannotChangePrefixes,2778,1"
        metamod_mg, metamod_m, metamod_ilvl = metamod_string.split(",")

    G = nx.DiGraph()

    # Add Block Node
    block_mg, block_mg_weight = get_blocking_affix(p_mg_weights, p_craftable) if req_affix_type == 'prefix' else get_blocking_affix(s_mg_weights, s_craftable)
    block_m = "block"
    block_ilvl = "0"
    block_string = f"{block_mg},{block_m},{block_ilvl}"
    G.add_node(
        block_string,
        affixes=1,
    )

    # First Slam
    for mod_group, mods in mod_groups.items():
        if mod_group == block_mg:
            continue
        for mod, details in mods.items():
            for detail in details:
                first_slam_string = f"{mod_group},{mod},{detail['ilvl']}"
                block_slam_string = f"{block_string}|{first_slam_string}"
                G.add_edge(
                    block_string,
                    block_slam_string,
                    weight=(detail['weight'] / (total_weight - block_mg_weight)),
                    cost=[2, 0, 0, 0]
                )
                G.nodes[block_slam_string]['affixes'] = 2

    # Second Slam
    for node in dict(G.nodes):  # make a copy since we're going to be adding nodes to our Graph as we go
        if G.nodes[node]['affixes'] != 2:
            continue

        block_string, first_slam_string = node.split("|")
        first_slam_mg = first_slam_string.split(",")[0]
        first_slam_mg_weight = mg_weights[first_slam_mg]

        for mod_group, mods in mod_groups.items():
            if mod_group in [block_mg, first_slam_mg]:
                continue
            for mod, details in mods.items():
                for detail in details:
                    second_slam_string = f"{mod_group},{mod},{detail['ilvl']}"
                    # append the first and second slam string to the block string in alphabetical order
                    block_slam_slam_string = f"{block_string}|{'|'.join(sorted([first_slam_string, second_slam_string]))}"
                    G.add_edge(
                        node,
                        block_slam_slam_string,
                        weight=(detail['weight'] / (total_weight - block_mg_weight - first_slam_mg_weight)),
                        cost=[2, 0, 0, 0]
                    )
                    G.nodes[block_slam_slam_string]['affixes'] = 3

    # Metacrafting
    for node in dict(G.nodes):
        if G.nodes[node]['affixes'] != 3:
            continue

        block_string, first_slam_string, second_slam_string = node.split("|")
        first_slam_mg, first_slam_m, first_slam_ilvl = first_slam_string.split(",")
        second_slam_mg, second_slam_m, second_slam_ilvl = second_slam_string.split(",")

        # Check if either slam is wanted in any required mod
        required_first = False
        required_second = False
        required_combo = False
        for mod in required_mods:
            mg, m, ilvl = mod.split(",")
            if m == first_slam_m and int(ilvl) <= int(first_slam_ilvl):
                required_first = True
            if m == second_slam_m and int(ilvl) <= int(second_slam_ilvl):
                required_second = True

        if required_first and required_second:
            for combo in required_combos:
                first_combo, second_combo = combo.split("|")
                if first_combo > second_combo:
                    first_combo, second_combo = second_combo, first_combo
                if first_slam_string > second_slam_string:
                    raise

                first_combo_mg, first_combo_m, first_combo_ilvl = first_combo.split(",")
                second_combo_mg, second_combo_m, second_combo_ilvl = second_combo.split(",")

                # since all mods are alphabetically sorted we can just see if firsts match and seconds match
                if (first_combo_m == first_slam_m and int(first_combo_ilvl) <= int(first_slam_ilvl) and
                        second_combo_m == second_slam_m and int(second_combo_ilvl) <= int(second_slam_ilvl)):
                    required_combo = True

        if required_combo:
            # GG combo, terminal node
            G.nodes[node]['terminal'] = True
        elif required_first and required_second:
            # We have two good mods that have no synergy, we want to annul any one of them
            metamod_node = f"{metamod_string}|{first_slam_string}|{second_slam_string}"
            G.add_edge(
                node,
                metamod_node,
                cost=[2, 0, 1, 0],
                weight=1
            )

            G.add_edge(
                metamod_node,
                metamod_node,
                cost=[2, 1, 0, 0],
                weight=1/3
            )
            G.add_edge(
                metamod_node,
                f"{block_string}|{first_slam_string}",
                cost=[0, 1, 1, 1],
                weight=1/3
            )
            G.add_edge(
                metamod_node,
                f"{block_string}|{second_slam_string}",
                cost=[0, 1, 1, 1],
                weight=1/3
            )
        elif required_first or required_second:
            # We have exactly one good mod we want to save
            metamod_node = f"{metamod_string}|{first_slam_string}|{second_slam_string}"
            G.add_edge(
                node,
                metamod_node,
                cost=[2, 0, 1, 0],
                weight=1
            )

            G.add_edge(
                metamod_node,
                metamod_node,
                cost=[1, 1, 0, 0],
                weight=1/3
            )
            G.add_edge(
                metamod_node,
                f"{block_string}",
                cost=[0, 1, 1, 1],
                weight=1/3
            )
            if required_first:
                G.add_edge(
                    metamod_node,
                    f"{block_string}|{first_slam_string}",
                    cost=[0, 1, 1, 1],
                    weight=1/3
                )
            else:
                G.add_edge(
                    metamod_node,
                    f"{block_string}|{second_slam_string}",
                    cost=[0, 1, 1, 1],
                    weight=1/3
                )
        else:
            # We have no good mods
            G.add_edge(
                node,
                block_string,
                cost=[2, 0, 2, 1],
                weight=1
            )

    # Vectorize all costs
    for src_node, dst_node in G.edges:
        cost = G.edges[src_node, dst_node]['cost']
        if len(cost) == 3:
            cost += [0]
        G.edges[src_node, dst_node]['cost'] = np.array(cost, dtype=float)

    for src_node in G:
        total = 0
        for dst_node in nx.neighbors(G, src_node):
            total += G.edges[src_node, dst_node]['weight']
        if abs(total - 1) > 0.0000000000001 and 'terminal' not in G.nodes[src_node]:
            print("WARNING: Node ", src_node, " sums to probability ", total)

    print("Terminal Nodes:")
    for node in G:
        if 'terminal' in G.nodes[node]:
            print(node)

    print(len(G), "nodes")
    print(len(G.edges()), "edges")
    G.start = block_string
    return G

def time2goal(G, epsilon=0.01):
    state_vec = defaultdict(lambda: 0)
    cost_vec = defaultdict(lambda: np.zeros(4))
    # ie state_vec['Block'] = 1
    state_vec[G.start] = 1

    success = [defaultdict(lambda: 0)]
    success_cost = [defaultdict(lambda: np.zeros(4))]

    while sum(state_vec.values()) > epsilon:
        state_vec, cost_vec = transition(G, state_vec, cost_vec)
        success_at_current_step = {}
        cost_at_current_step = {}
        for node in G.nodes:
            if 'terminal' in G.nodes[node]:
                success_at_current_step[node] = state_vec[node]
                cost_at_current_step[node] = cost_vec[node]
                state_vec[node] = 0
                cost_vec[node] = np.zeros(4)
        success.append(success_at_current_step)
        success_cost.append(cost_at_current_step)

    return report_time(success, success_cost)

def transition(G, state_vec, cost_vec):
    new_state_vec = defaultdict(lambda: 0)
    new_cost_vec = defaultdict(lambda: 0)
    for src_node, dst_node in G.edges():
        weight = state_vec[src_node] * G.edges[src_node, dst_node]['weight']
        new_state_vec[dst_node] += weight
        new_cost_vec[dst_node] += weight * cost_vec[src_node]  # previous costs
        new_cost_vec[dst_node] += weight * G.edges[src_node, dst_node]['cost']  # current transition cost
    for node in G:
        if new_state_vec[node] > 0:
            new_cost_vec[node] = new_cost_vec[node] / new_state_vec[node]

    return new_state_vec, new_cost_vec


def report_time(success, success_cost):
    combined_success = []
    for d in success:
        step_success = 0
        for k in d:
            step_success += d[k]
        combined_success.append(step_success)

    cumulative_success = np.cumsum(combined_success).tolist()
    distribution = reduce(lambda x, y: {k: x[k] + y[k] for k in y}, success)

    # Convert ndarrays to lists to make the json serializable
    for costs in success_cost:
        for k, cost in costs.items():
            costs[k] = cost.tolist()

    report = {
        'cumulative_success': cumulative_success,
        'success_cost': success_cost,
        'distribution': distribution
    }

    return report

