import numpy as np
import itertools
import math
import logging
from functools import lru_cache
from collections import Counter

from ..core.alignment import align_chains, format_alignment
from .operations_nocy import residue_distances
from .constants import * 

@lru_cache
def get_residue_distances(chain1, chain2, what, all_atom=True):
    if all_atom:
        # how many atoms per aligned amino acid
        n_atoms_per_res_chain1 = list_atoms_per_residue(chain1, what)
        n_atoms_per_res_chain2 = list_atoms_per_residue(chain2, what)
        model_A_atoms = np.asarray(
            [atom.coord for res in chain1 for atom in res.get_atoms()]
        )
        model_B_atoms = np.asarray(
            [atom.coord for res in chain2 for atom in res.get_atoms()]
        )

    else:  # distances were already between CBs only
        model_A_atoms = np.asarray(
            [
                res["CB"].get_coord() if "CB" in res else res["CA"].get_coord()
                for res in chain1
            ]
        )
        model_B_atoms = np.asarray(
            [
                res["CB"].get_coord() if "CB" in res else res["CA"].get_coord()
                for res in chain2
            ]
        )

        n_atoms_per_res_chain1 = np.ones(model_A_atoms.shape[0]).astype(int)
        n_atoms_per_res_chain2 = np.ones(model_B_atoms.shape[0]).astype(int)

    model_res_distances = residue_distances(
        model_A_atoms, model_B_atoms, n_atoms_per_res_chain1, n_atoms_per_res_chain2
    )
    return model_res_distances

def get_interacting_pairs(distances, threshold):
    interacting_pairs = np.nonzero(np.asarray(distances) < threshold)
    return tuple(interacting_pairs[0]), tuple(interacting_pairs[1])

@lru_cache
def subset_atoms(
    mod_chain,
    ref_chain,
    atom_types,
    residue_subset=None,
    what="",
):
    mod_atoms = []
    ref_atoms = []

    mod_residues = [res for res in mod_chain]
    ref_residues = [res for res in ref_chain]

    # remove duplicate residues
    residue_subset = set(residue_subset) if residue_subset else range(len(mod_residues))

    for i in residue_subset:
        mod_res_atoms = list(mod_residues[i].get_atoms())
        ref_res_atoms = list(ref_residues[i].get_atoms())
        mod_res_atoms_ids = [atom.id for atom in mod_res_atoms]
        ref_res_atoms_ids = [atom.id for atom in ref_res_atoms]

        for atom_type in atom_types:
            try:
                mod_i = mod_res_atoms_ids.index(atom_type)
                ref_i = ref_res_atoms_ids.index(atom_type)
                mod_atoms += [mod_res_atoms[mod_i].coord]
                ref_atoms += [ref_res_atoms[ref_i].coord]
            except:
                continue

    return mod_atoms, ref_atoms

@lru_cache
def list_atoms_per_residue(chain, what):
    n_atoms_per_residue = []

    for residue in chain:
        # important to remove duplicate atoms (e.g. alternates) at this stage
        atom_ids = set([a.id for a in residue.get_unpacked_list()])
        n_atoms_per_residue.append(len(atom_ids))
    return np.array(n_atoms_per_residue).astype(int)



def create_graph(atom_list, atom_ids):
    import networkx as nx

    G = nx.Graph()

    for i, atom_i in enumerate(atom_list):
        cr_i = COVALENT_RADIUS[atom_ids[i]]
        for j, atom_j in enumerate(atom_list):
            cr_j = COVALENT_RADIUS[atom_ids[j]]
            distance = np.linalg.norm(atom_i - atom_j)
            threshold = (cr_i + cr_j + BOND_TOLERANCE) if i != j else 1
            if distance < threshold:  # Adjust threshold as needed
                G.add_edge(i, j)

    return G


def group_chains(
    query_structure, ref_structure, query_chains, ref_chains, allowed_mismatches=0
):
    reverse_map = False
    mismatch_dict = {}  # for diagnostics
    # this might happen e.g. when modelling only part of a large homomer
    if len(query_chains) < len(ref_chains):
        query_structure, ref_structure = ref_structure, query_structure
        query_chains, ref_chains = ref_chains, query_chains
        reverse_map = True

    alignment_targets = itertools.product(query_chains, ref_chains)
    chain_clusters = {chain: [] for chain in ref_chains}

    for query_chain, ref_chain in alignment_targets:
        try:
            qc = query_structure[query_chain]
        except KeyError:
            logging.error(
                f"""The specified model chain {query_chain} is not found in the PDB structure.
This is possibly due to using the wrong chain identifier in --mapping,
or forgetting to specify --small_molecule if this is a HETATM chain.
If working with mmCIF files, make sure you use the right chain identifier.
            """
            )
            print(traceback.format_exc())
            sys.exit(1)
        try:
            rc = ref_structure[ref_chain]
        except KeyError:
            logging.error(
                f"""The specified native chain {ref_chain} is not found in the PDB structure.
This is possibly due to using the wrong chain identifier in --mapping,
or forgetting to specify --small_molecule if this is a HETATM chain.
If working with mmCIF files, make sure you use the right chain identifier.
            """
            )
        het_qc = qc.is_het
        het_rc = rc.is_het

        if het_qc is None and het_rc is None:
            aln = align_chains(
                qc,
                rc,
                use_numbering=False,
            )
            alignment = format_alignment(aln)
            n_mismatches = alignment["matches"].count(".")

            if 0 < n_mismatches < 10:
                mismatch_dict[(query_chain, ref_chain)] = n_mismatches

            if n_mismatches <= allowed_mismatches:
                # 100% sequence identity, 100% coverage of native sequence in model sequence
                chain_clusters[ref_chain].append(query_chain)
        elif het_qc and het_rc and het_qc == het_rc:
            chain_clusters[ref_chain].append(query_chain)
    chains_without_match = [
        chain for chain in chain_clusters if not chain_clusters[chain]
    ]

    if mismatch_dict:
        logging.warning(
            f"""Some chains have a limited number of sequence mismatches and are treated as non-homologous. 
Try increasing the --allowed_mismatches for the following: {", ".join(f"Model chain {c[1]}, native chain {c[0]}: {m} mismatches" for c, m in mismatch_dict.items())}
if they should be treated as homologous."""
        )

    if chains_without_match:
        logging.error(
            f"For chains {chains_without_match} no identical corresponding chain was found between in the native."
        )
        sys.exit(1)

    return chain_clusters, reverse_map


def format_mapping(mapping_str, small_molecule=None):
    mapping = dict()
    model_chains = None
    native_chains = None
    if not mapping_str:
        return mapping, model_chains, native_chains

    model_mapping, native_mapping = mapping_str.split(":")
    if not native_mapping:
        logging.error(
            "When using --mapping, native chains must be set (e.g. ABC:ABC or :ABC)"
        )
        sys.exit()
    else:
        # :ABC or *:ABC only use those natives chains, permute model chains
        if not model_mapping or model_mapping == "*":
            native_chains = [chain for chain in native_mapping]
        elif len(model_mapping) == len(native_mapping):
            # ABC*:ABC* fix the first part of the mapping, try all other combinations
            mapping = {
                nm: mm
                for nm, mm in zip(native_mapping, model_mapping)
                if nm != "*" and mm != "*"
            }
            if model_mapping[-1] != "*" and native_mapping[-1] != "*":
                # ABC:ABC use the specific mapping
                model_chains = [chain for chain in model_mapping]
                native_chains = [chain for chain in native_mapping]
    return mapping, model_chains, native_chains


def format_mapping_string(chain_mapping):
    chain1 = ""
    chain2 = ""

    # Sorting might change LRMSD since the definition of receptor/ligand for equal length depends on order
    mapping = [(b, a) for a, b in chain_mapping.items()]
    for (
        model_chain,
        native_chain,
    ) in mapping:
        chain1 += model_chain
        chain2 += native_chain

    return f"{chain1}:{chain2}"


def product_without_dupl(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [
            x + [y] for x in result for y in pool if y not in x
        ]  # here we added condition

    for prod in result:
        yield tuple(prod)


def count_chain_combinations(chain_clusters):
    try:
        clusters = [tuple(li) for li in chain_clusters.values()]
        number_of_combinations = np.prod(
            [
                int(math.factorial(len(a)) / math.factorial(len(a) - b))
                for a, b in Counter(clusters).items()
            ]
        )
    except ValueError:
        logging.error(
            """Couldn't find a match between each model-native chain specified in the mapping.
Make sure that all chains in your model have a homologous chain in the native, or specify the right subset of chains with --mapping"""
        )
        sys.exit()
    return number_of_combinations


def get_all_chain_maps(
    chain_clusters,
    initial_mapping,
    reverse_map,
    model_chains_to_combo,
    native_chains_to_combo,
):
    all_mappings = product_without_dupl(
        *[cluster for cluster in chain_clusters.values() if cluster]
    )
    for mapping in all_mappings:
        chain_map = {key: value for key, value in initial_mapping.items()}
        if reverse_map:
            chain_map.update(
                {
                    mapping[i]: model_chain
                    for i, model_chain in enumerate(model_chains_to_combo)
                }
            )
        else:
            chain_map.update(
                {
                    native_chain: mapping[i]
                    for i, native_chain in enumerate(native_chains_to_combo)
                }
            )
        yield chain_map


def get_chain_map_from_dockq(result):
    chain_map = {}
    for ch1, ch2 in result:
        chain_map[ch1] = result[ch1, ch2]["chain1"]
        chain_map[ch2] = result[ch1, ch2]["chain2"]
    return chain_map
