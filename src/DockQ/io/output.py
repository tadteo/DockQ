
def print_results(
    info, short=False, verbose=False, capri_peptide=False, small_molecule=False
):

    score = (
        "DockQ-small_molecules"
        if small_molecule
        else "DockQ-capri_peptide"
        if capri_peptide
        else "DockQ"
    )
    if short:
        print(
            f"Total {score} over {len(info['best_result'])} native interfaces: {info['GlobalDockQ']:.3f} with {info['best_mapping_str']} model:native mapping"
        )
        for chains, results in info["best_result"].items():
            reported_measures = (
                [
                    "DockQ",
                    "iRMSD",
                    "LRMSD",
                    "fnat",
                    "fnonnat",
                    "F1",
                    "clashes",
                ]
                if not results["is_het"]
                else ["LRMSD"]
            )
            hetname = f" ({results['is_het']})" if results["is_het"] else ""
            score_str = " ".join(
                [
                    f"{item} {results[item]:.3f}"
                    if item != "clashes"
                    else f"{item} {results[item]}"
                    for item in reported_measures
                ]
            )
            print(
                f"{score_str} mapping {results['chain1']}{results['chain2']}:{chains[0]}{chains[1]}{hetname} {info['model']} {results['chain1']} {results['chain2']} -> {info['native']} {chains[0]} {chains[1]}"
            )
    else:
        print_header(verbose, capri_peptide)
        print(f"Model  : {info['model']}")
        print(f"Native : {info['native']}")
        print(
            f"Total {score} over {len(info['best_result'])} native interfaces: {info['GlobalDockQ']:.3f} with {info['best_mapping_str']} model:native mapping"
        )
        for chains, results in info["best_result"].items():
            reported_measures = (
                [
                    "DockQ",
                    "iRMSD",
                    "LRMSD",
                    "fnat",
                    "fnonnat",
                    "F1",
                    "clashes",
                ]
                if not results["is_het"]
                else ["LRMSD"]
            )
            hetname = f" ({results['is_het']})" if results["is_het"] else ""
            print(f"Native chains: {chains[0]}, {chains[1]}{hetname}")
            print(f"\tModel chains: {results['chain1']}, {results['chain2']}")
            print(
                "\n".join(
                    [
                        f"\t{item}: {results[item]:.3f}"
                        if item != "clashes"
                        else f"\t{item}: {results[item]}"
                        for item in reported_measures
                    ]
                )
            )


def print_header(verbose=False, capri_peptide=False):
    reference = (
        "*   Ref: Mirabello and Wallner, 'DockQ v2: Improved automatic  *\n"
        "*   quality measure for protein multimers, nucleic acids       *\n"
        "*   and small molecules'                                       *\n"
        "*                                                              *\n"
        "*   For comments, please email: bjorn.wallner@.liu.se          *"
    )

    header = (
        "****************************************************************\n"
        "*                       DockQ                                  *\n"
        "*   Docking scoring for biomolecular models                    *\n"
        "*   DockQ score legend:                                        *\n"
        "*    0.00 <= DockQ <  0.23 - Incorrect                         *\n"
        "*    0.23 <= DockQ <  0.49 - Acceptable quality                *\n"
        "*    0.49 <= DockQ <  0.80 - Medium quality                    *\n"
        "*            DockQ >= 0.80 - High quality                      *"
    )

    if verbose:
        notice = (
            "*   For the record:                                            *\n"
            f"*   Definition of contact <{'5A' if not capri_peptide else '4A'} (Fnat)                           *\n"
            f"*   Definition of interface <{'10A all heavy atoms (iRMSD)      ' if not capri_peptide else '8A CB (iRMSD)                    '} *\n"
            "****************************************************************"
        )
    else:
        notice = "****************************************************************"

    print(header)
    print(reference)
    print(notice)


