import gzip
from ..core.structure import Structure
from ..utils.parsers import PDBParser, MMCIFParser

def load_PDB(path, chains=[], small_molecule=False, n_model=0):
    try:
        pdb_parser = PDBParser(QUIET=True)
        model = pdb_parser.get_structure(
            "-",
            (gzip.open if path.endswith(".gz") else open)(path, "rt"),
            chains=chains,
            parse_hetatms=small_molecule,
            model_number=n_model,
        )
    except Exception:
        pdb_parser = MMCIFParser(QUIET=True)
        model = pdb_parser.get_structure(
            "-",
            (gzip.open if path.endswith(".gz") else open)(path, "rt"),
            chains=chains,
            parse_hetatms=small_molecule,
            auth_chains=not small_molecule,
            model_number=n_model,
        )
    model.id = path
    return model
