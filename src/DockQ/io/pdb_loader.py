import gzip
from ..core.structure import Structure
from ..utils.parsers import PDBParser, MMCIFParser
from Bio.PDB.Structure import Structure as BioStructure

def load_PDB(input_data, chains=[], small_molecule=False, n_model=0):
    """
    Load a PDB or mmCIF structure from a file or a Bio.PDB.Structure object.

    Args:
        input_data (str or Bio.PDB.Structure): File path or Structure object to load.
        chains (list): List of chain IDs to load. If empty, load all chains.
        small_molecule (bool): If True, parse HETATM records (for small molecules).
        n_model (int): Model number to load from multi-model files (default: 0).

    Returns:
        Structure: A Structure object containing the loaded data.

    Raises:
        ValueError: If input_data is neither a file path nor a Bio.PDB.Structure object.
    """
    if isinstance(input_data, str):  # It's a file path
        try:
            # First, try to parse as PDB format
            pdb_parser = PDBParser(QUIET=True)
            model = pdb_parser.get_structure(
                "-",
                (gzip.open if input_data.endswith(".gz") else open)(input_data, "rt"),
                chains=chains,
                parse_hetatms=small_molecule,
                model_number=n_model,
            )
        except Exception:
            # If PDB parsing fails, try mmCIF format
            pdb_parser = MMCIFParser(QUIET=True)
            model = pdb_parser.get_structure(
                "-",
                (gzip.open if input_data.endswith(".gz") else open)(input_data, "rt"),
                chains=chains,
                parse_hetatms=small_molecule,
                auth_chains=not small_molecule,
                model_number=n_model,
            )
        model.id = input_data  # Set the model ID to the input file path
    elif isinstance(input_data, BioStructure):  # It's already a structure object
        model = input_data[n_model]  # Get the specified model
        model.id = "input_structure"
    else:
        raise ValueError("Input must be either a file path or a Bio.PDB.Structure object")
    
    return Structure(model)  # Return a new Structure object initialized with the loaded model
