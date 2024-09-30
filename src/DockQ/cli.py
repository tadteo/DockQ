#!/usr/bin/env python

import sys
import json
import logging
import itertools
from functools import partial
from argparse import ArgumentParser
from parallelbar import progress_map

from .core.scoring import calc_DockQ, calc_sym_corrected_lrmsd, run_on_all_native_interfaces
from .io.pdb_loader import load_PDB
from .io.output import print_results
from .utils.helpers import format_mapping, group_chains, get_all_chain_maps, count_chain_combinations, format_mapping_string


def parse_args():
    parser = ArgumentParser(
        description="DockQ - Quality measure for \
        protein-protein docking models"
    )
    parser.add_argument("model", metavar="<model>", type=str, help="Path to model file")
    parser.add_argument(
        "native", metavar="<native>", type=str, help="Path to native file"
    )
    parser.add_argument(
        "--capri_peptide",
        default=False,
        action="store_true",
        help="use version for capri_peptide \
        (DockQ cannot not be trusted for this setting)",
    )
    parser.add_argument(
        "--small_molecule",
        help="If the docking pose of a small molecule should be evaluated",
        action="store_true",
    )
    parser.add_argument(
        "--short", default=False, action="store_true", help="Short output"
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="out.json",
        help="Write outputs to a chosen json file",
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="Do not align native and model using sequence alignments, but use the numbering of residues instead",
    )
    parser.add_argument(
        "--n_cpu",
        default=8,
        type=int,
        metavar="CPU",
        help="Number of cores to use",
    )
    parser.add_argument(
        "--max_chunk",
        default=512,
        type=int,
        metavar="CHUNK",
        help="Maximum size of chunks given to the cores, actual chunksize is min(max_chunk,combos/cpus)",
    )
    parser.add_argument(
        "--allowed_mismatches",
        default=0,
        type=int,
        help="Number of allowed mismatches when mapping model sequence to native sequence.",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        metavar="MODELCHAINS:NATIVECHAINS",
        help="""Specify a chain mapping between model and native structure.
            If the native contains two chains "H" and "L"
            while the model contains two chains "A" and "B",
            and chain A is a model of native chain
            H and chain B is a model of native chain L,
            the flag can be set as: '--mapping AB:HL'.
            This can also help limit the search to specific native interfaces.
            For example, if the native is a tetramer (ABCD) but the user is only interested
            in the interface between chains B and C, the flag can be set as: '--mapping :BC'
            or the equivalent '--mapping *:BC'.""",
    )

    return parser.parse_args()




# @profile
def main():
    args = parse_args()

    initial_mapping, model_chains, native_chains = format_mapping(
        args.mapping, args.small_molecule
    )
    model_structure = load_PDB(
        args.model, chains=model_chains, small_molecule=args.small_molecule
    )
    native_structure = load_PDB(
        args.native, chains=native_chains, small_molecule=args.small_molecule
    )
    # check user-given chains are in the structures
    model_chains = [c.id for c in model_structure] if not model_chains else model_chains
    native_chains = (
        [c.id for c in native_structure] if not native_chains else native_chains
    )

    if len(model_chains) < 2 or len(native_chains) < 2:
        print("Need at least two chains in the two inputs\n")
        sys.exit()

    # permute chains and run on a for loop
    best_dockq = -1
    best_result = None
    best_mapping = None

    model_chains_to_combo = [
        mc for mc in model_chains if mc not in initial_mapping.values()
    ]
    native_chains_to_combo = [
        nc for nc in native_chains if nc not in initial_mapping.keys()
    ]

    chain_clusters, reverse_map = group_chains(
        model_structure,
        native_structure,
        model_chains_to_combo,
        native_chains_to_combo,
        args.allowed_mismatches,
    )
    chain_maps = get_all_chain_maps(
        chain_clusters,
        initial_mapping,
        reverse_map,
        model_chains_to_combo,
        native_chains_to_combo,
    )

    num_chain_combinations = count_chain_combinations(chain_clusters)
    # copy iterator to use later
    chain_maps, chain_maps_ = itertools.tee(chain_maps)

    low_memory = num_chain_combinations > 100
    run_chain_map = partial(
        run_on_all_native_interfaces,
        model_structure,
        native_structure,
        no_align=args.no_align,
        capri_peptide=args.capri_peptide,
        low_memory=low_memory,
    )

    if num_chain_combinations > 1:
        cpus = min(num_chain_combinations, args.n_cpu)
        chunk_size = min(args.max_chunk, max(1, num_chain_combinations // cpus))

        # for large num_chain_combinations it should be possible to divide the chain_maps in chunks
        result_this_mappings = progress_map(
            run_chain_map,
            chain_maps,
            total=num_chain_combinations,
            n_cpu=cpus,
            chunk_size=chunk_size,
        )

        for chain_map, (result_this_mapping, total_dockq) in zip(
            chain_maps_, result_this_mappings
        ):

            if total_dockq > best_dockq:
                best_dockq = total_dockq
                best_result = result_this_mapping
                best_mapping = chain_map

        if low_memory:  # retrieve the full output by rerunning the best chain mapping
            best_result, total_dockq = run_on_all_native_interfaces(
                model_structure,
                native_structure,
                chain_map=best_mapping,
                no_align=args.no_align,
                capri_peptide=args.capri_peptide,
                low_memory=False,
            )

    else:  # skip multi-threading for single jobs (skip the bar basically)
        best_mapping = next(chain_maps)
        best_result, best_dockq = run_chain_map(best_mapping)

    if not best_result:
        logging.error(
            "Could not find interfaces in the native model. Please double check the inputs or select different chains with the --mapping flag."
        )
        sys.exit(1)

    info = dict()
    info["model"] = args.model
    info["native"] = args.native
    info["best_dockq"] = best_dockq
    info["best_result"] = best_result
    info["GlobalDockQ"] = best_dockq / len(best_result)
    info["best_mapping"] = best_mapping
    info["best_mapping_str"] = f"{format_mapping_string(best_mapping)}"

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(info, fp)

    print_results(
        info, args.short, args.verbose, args.capri_peptide, args.small_molecule
    )


if __name__ == "__main__":
    main()
