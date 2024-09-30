import numpy as np
from functools import lru_cache
import itertools

from Bio.SVDSuperimposer import SVDSuperimposer
from ..utils.constants import *
from ..utils.helpers import get_residue_distances, get_interacting_pairs, subset_atoms, create_graph
from ..utils.operations_nocy import get_fnat_stats
from .alignment import align_chains, format_alignment, get_aligned_residues

# @profile
def calc_DockQ(
    sample_chains,
    ref_chains,
    alignments,
    capri_peptide=False,
    low_memory=False,
):

    fnat_threshold = FNAT_THRESHOLD if not capri_peptide else FNAT_THRESHOLD_PEPTIDE
    interface_threshold = (
        INTERFACE_THRESHOLD if not capri_peptide else INTERFACE_THRESHOLD_PEPTIDE
    )

    # total number of native contacts is calculated on untouched native structure
    ref_res_distances = get_residue_distances(ref_chains[0], ref_chains[1], "ref")
    nat_total = np.nonzero(np.asarray(ref_res_distances) < fnat_threshold ** 2)[
        0
    ].shape[0]

    if nat_total == 0:
        # if the native has no interface between the two chain groups
        # nothing to do here
        return None

    aligned_sample_1, aligned_ref_1 = get_aligned_residues(
        sample_chains[0], ref_chains[0], alignments[0]
    )
    aligned_sample_2, aligned_ref_2 = get_aligned_residues(
        sample_chains[1], ref_chains[1], alignments[1]
    )

    sample_res_distances = get_residue_distances(
        aligned_sample_1, aligned_sample_2, "sample"
    )

    if ref_res_distances.shape != sample_res_distances.shape:
        ref_res_distances = get_residue_distances(aligned_ref_1, aligned_ref_2, "ref")

    assert (
        sample_res_distances.shape == ref_res_distances.shape
    ), f"Native and model have incompatible sizes ({sample_res_distances.shape} != {ref_res_distances.shape})"

    nat_correct, nonnat_count, _, model_total = get_fnat_stats(
        sample_res_distances, ref_res_distances, threshold=fnat_threshold
    )

    # avoids divide by 0 errors
    fnat = nat_total and nat_correct / nat_total or 0
    fnonnat = model_total and nonnat_count / model_total or 0

    if capri_peptide:
        ref_res_distances = get_residue_distances(
            aligned_ref_1, aligned_ref_2, "ref", all_atom=False
        )
    # Get interfacial atoms from reference, and corresponding atoms from sample
    interacting_pairs = get_interacting_pairs(
        # working with squared thresholds to avoid using sqrt
        ref_res_distances,
        threshold=interface_threshold ** 2,
    )

    sample_interface_atoms1, ref_interface_atoms1 = subset_atoms(
        aligned_sample_1,
        aligned_ref_1,
        atom_types=BACKBONE_ATOMS,
        residue_subset=interacting_pairs[0],
    )
    sample_interface_atoms2, ref_interface_atoms2 = subset_atoms(
        aligned_sample_2,
        aligned_ref_2,
        atom_types=BACKBONE_ATOMS,
        residue_subset=interacting_pairs[1],
    )

    sample_interface_atoms = np.asarray(
        sample_interface_atoms1 + sample_interface_atoms2
    )
    ref_interface_atoms = np.asarray(ref_interface_atoms1 + ref_interface_atoms2)

    super_imposer = SVDSuperimposer()
    super_imposer.set(sample_interface_atoms, ref_interface_atoms)
    super_imposer.run()
    irms = super_imposer.get_rms()

    # assign which chains constitute the receptor, ligand
    ref_group1_size = len(ref_chains[0])
    ref_group2_size = len(ref_chains[1])
    receptor_chains = (
        (aligned_ref_1, aligned_sample_1)
        if ref_group1_size > ref_group2_size
        else (aligned_ref_2, aligned_sample_2)
    )
    ligand_chains = (
        (aligned_ref_1, aligned_sample_1)
        if ref_group1_size <= ref_group2_size
        else (aligned_ref_2, aligned_sample_2)
    )
    class1, class2 = (
        ("receptor", "ligand")
        if ref_group1_size > ref_group2_size
        else ("ligand", "receptor")
    )

    receptor_atoms_native, receptor_atoms_sample = subset_atoms(
        receptor_chains[0],
        receptor_chains[1],
        atom_types=BACKBONE_ATOMS,
        what="receptor",
    )
    ligand_atoms_native, ligand_atoms_sample = subset_atoms(
        ligand_chains[0], ligand_chains[1], atom_types=BACKBONE_ATOMS, what="ligand"
    )
    # Set to align on receptor
    super_imposer.set(
        np.asarray(receptor_atoms_native), np.asarray(receptor_atoms_sample)
    )
    super_imposer.run()

    rot, tran = super_imposer.get_rotran()
    rotated_sample_atoms = np.dot(np.asarray(ligand_atoms_sample), rot) + tran

    lrms = super_imposer._rms(
        np.asarray(ligand_atoms_native), rotated_sample_atoms
    )  # using the private _rms function which does not superimpose

    info = {}
    F1 = f1(nat_correct, nonnat_count, nat_total)
    info["DockQ"] = dockq_formula(fnat, irms, lrms)
    if low_memory:
        return info

    info["F1"] = F1
    info["iRMSD"] = irms
    info["LRMSD"] = lrms
    info["fnat"] = fnat
    info["nat_correct"] = nat_correct
    info["nat_total"] = nat_total

    info["fnonnat"] = fnonnat
    info["nonnat_count"] = nonnat_count
    info["model_total"] = model_total
    info["clashes"] = np.nonzero(
        np.asarray(sample_res_distances) < CLASH_THRESHOLD ** 2
    )[0].shape[0]
    info["len1"] = ref_group1_size
    info["len2"] = ref_group2_size
    info["class1"] = class1
    info["class2"] = class2
    info["is_het"] = False

    return info

def calc_sym_corrected_lrmsd(
    sample_chains,
    ref_chains,
    alignments,
):
    import networkx as nx

    is_het_sample_0 = bool(sample_chains[0].is_het)
    is_het_sample_1 = bool(sample_chains[1].is_het)

    if is_het_sample_0 and not is_het_sample_1:
        sample_ligand = sample_chains[0]
        sample_receptor = sample_chains[1]
        ref_ligand = ref_chains[0]
        ref_receptor = ref_chains[1]
        receptor_alignment = alignments[1]
    elif not is_het_sample_0 and is_het_sample_1:
        sample_ligand = sample_chains[1]
        sample_receptor = sample_chains[0]
        ref_ligand = ref_chains[1]
        ref_receptor = ref_chains[0]
        receptor_alignment = alignments[0]
    else:
        return  # both ligands, no lrmsd

    ref_res_distances = get_residue_distances(ref_receptor, ref_ligand, "ref")
    receptor_interface, _ = get_interacting_pairs(
        # working with squared thresholds to avoid using sqrt
        ref_res_distances,
        threshold=INTERFACE_THRESHOLD ** 2,
    )
    if not receptor_interface:
        return
    aligned_sample_receptor, aligned_ref_receptor = get_aligned_residues(
        sample_receptor, ref_receptor, receptor_alignment
    )

    sample_interface_atoms, ref_interface_atoms = subset_atoms(
        aligned_sample_receptor,
        aligned_ref_receptor,
        atom_types=BACKBONE_ATOMS,
        residue_subset=receptor_interface,
        what="receptor",
    )

    sample_ligand_atoms_ids = [atom.id for atom in sample_ligand.get_atoms()]
    sample_ligand_atoms_ele = [atom.element for atom in sample_ligand.get_atoms()]

    ref_ligand_atoms_ids = [atom.id for atom in ref_ligand.get_atoms()]
    ref_ligand_atoms_ele = [atom.element for atom in ref_ligand.get_atoms()]

    sample_ligand_atoms = np.array(
        [
            atom.coord
            for atom in sample_ligand.get_atoms()
            if atom.id in ref_ligand_atoms_ids
        ]
    )
    ref_ligand_atoms = np.array(
        [
            atom.coord
            for atom in ref_ligand.get_atoms()
            if atom.id in sample_ligand_atoms_ids
        ]
    )

    # Set to align on receptor interface
    super_imposer = SVDSuperimposer()
    super_imposer.set(
        np.asarray(ref_interface_atoms), np.asarray(sample_interface_atoms)
    )
    super_imposer.run()
    rot, tran = super_imposer.get_rotran()

    sample_rotated_ligand_atoms = np.dot(sample_ligand_atoms, rot) + tran

    sample_graph = create_graph(sample_ligand_atoms, sample_ligand_atoms_ele)
    ref_graph = create_graph(ref_ligand_atoms, ref_ligand_atoms_ele)

    min_lrms = float("inf")
    best_mapping = None

    for isomorphism in nx.vf2pp_all_isomorphisms(sample_graph, ref_graph):
        model_i = list(isomorphism.keys())
        native_i = list(isomorphism.values())

        lrms = super_imposer._rms(
            sample_rotated_ligand_atoms[model_i], ref_ligand_atoms[native_i]
        )
        if lrms < min_lrms:
            best_mapping = isomorphism
            min_lrms = lrms
    dockq = dockq_formula(0, 0, min_lrms)
    info = {
        "DockQ": dockq,
        "LRMSD": min_lrms,
        "mapping": best_mapping,
        "is_het": sample_ligand.is_het,
    }
    return info

def dockq_formula(fnat, irms, lrms):
    return (
        float(fnat)
        + 1 / (1 + (irms / 1.5) * (irms / 1.5))
        + 1 / (1 + (lrms / 8.5) * (lrms / 8.5))
    ) / 3

def f1(tp, fp, p):
    return 2 * tp / (tp + fp + p)


@lru_cache
def run_on_chains(
    model_chains,
    native_chains,
    no_align=False,
    capri_peptide=False,
    small_molecule=True,
    low_memory=False,
):
    # realign each model chain against the corresponding native chain
    alignments = []
    for model_chain, native_chain in zip(model_chains, native_chains):
        aln = align_chains(
            model_chain,
            native_chain,
            use_numbering=no_align,
        )
        alignment = format_alignment(aln)
        alignments.append(tuple(alignment.values()))

    if not small_molecule:
        info = calc_DockQ(
            model_chains,
            native_chains,
            alignments=tuple(alignments),
            capri_peptide=capri_peptide,
            low_memory=low_memory,
        )
    else:
        info = calc_sym_corrected_lrmsd(
            model_chains,
            native_chains,
            alignments=tuple(alignments),
        )
    return info


def run_on_all_native_interfaces(
    model_structure,
    native_structure,
    chain_map={"A": "A", "B": "B"},
    no_align=False,
    capri_peptide=False,
    low_memory=False,
):
    """Given a native-model chain map, finds all non-null native interfaces
    and runs DockQ for each native-model pair of interfaces"""
    result_mapping = dict()
    native_chain_ids = list(chain_map.keys())

    for chain_pair in itertools.combinations(native_chain_ids, 2):
        native_chains = tuple([native_structure[chain] for chain in chain_pair])
        model_chains = tuple(
            [
                model_structure[chain]
                for chain in [chain_map[chain_pair[0]], chain_map[chain_pair[1]]]
            ]
        )

        small_molecule = native_chains[0].is_het or native_chains[1].is_het

        if len(set(model_chains)) < 2:
            continue
        if chain_pair[0] in chain_map and chain_pair[1] in chain_map:
            info = run_on_chains(
                model_chains,
                native_chains,
                no_align=no_align,
                capri_peptide=capri_peptide,
                small_molecule=small_molecule,
                low_memory=low_memory,
            )
            if info:
                info["chain1"], info["chain2"] = (
                    chain_map[chain_pair[0]],
                    chain_map[chain_pair[1]],
                )
                info["chain_map"] = chain_map  # diagnostics
                result_mapping["".join(chain_pair)] = info
    total_dockq = sum([result["DockQ"] for result in result_mapping.values()])
    return result_mapping, total_dockq


