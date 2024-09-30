from Bio import Align
from functools import lru_cache

@lru_cache
def align_chains(model_chain, native_chain, use_numbering=False):
    """
    Function to align two PDB structures. This can be done by sequence (default) or by
    numbering. If the numbering is used, then each residue number from the pdb structure
    is converted to a unique character. Then the two vectors of character are aligned
    as if they were two sequences
    """

    if use_numbering:
        model_numbering = []
        native_numbering = []

        for residue in model_chain.get_residues():
            resn = int(residue.id[1])
            model_numbering.append(resn)

        for residue in native_chain.get_residues():
            resn = int(residue.id[1])
            native_numbering.append(resn)
        # if the smallest resn is negative, it will be used to shift all numbers so they start from 0
        # the minimum offset is 45 to avoid including the "-" character that is reserved for gaps
        min_resn = max(45, -min(model_numbering + native_numbering))

        model_sequence = "".join([chr(resn + min_resn) for resn in model_numbering])
        native_sequence = "".join([chr(resn + min_resn) for resn in native_numbering])

    else:
        model_sequence = model_chain.sequence
        native_sequence = native_chain.sequence

    aligner = Align.PairwiseAligner()
    aligner.match = 5
    aligner.mismatch = 0
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -0.5
    aln = aligner.align(model_sequence, native_sequence)[0]
    return aln


def format_alignment(aln):
    alignment = {}
    try:
        alignment["seqA"] = aln[0, :]
        alignment["matches"] = "".join(
            [
                "|" if aa1 == aa2 else " " if (aa1 == "-" or aa2 == "-") else "."
                for aa1, aa2 in zip(aln[0, :], aln[1, :])
            ]
        )
        alignment["seqB"] = aln[1, :]
    except NotImplementedError:
        formatted_aln = aln.format().split("\n")
        alignment["seqA"] = formatted_aln[0]
        alignment["matches"] = formatted_aln[1]
        alignment["seqB"] = formatted_aln[2]

    return alignment

@lru_cache
def get_aligned_residues(chainA, chainB, alignment):
    aligned_resA = []
    aligned_resB = []
    resA = chainA.get_residues()
    resB = chainB.get_residues()

    if alignment[0] == alignment[2]:
        return tuple(resA), tuple(resB)

    for A, match, B in zip(*alignment):
        if A != "-":
            rA = next(resA)
        if B != "-":
            rB = next(resB)

        if match == "|":
            aligned_resA.append(rA)
            aligned_resB.append(rB)

    return tuple(aligned_resA), tuple(aligned_resB)

