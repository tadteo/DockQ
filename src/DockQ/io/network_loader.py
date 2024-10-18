import numpy as np
from Bio.PDB.Structure import Structure as BioStructure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from ..core.structure import Structure

def create_structure_from_network(chain_data, structure_id="network_output"):
    """
    Create a Structure object from network output data.

    This function takes a dictionary of chain data and constructs a Bio.PDB Structure
    object, which is then wrapped in a custom Structure class.

    Args:
        chain_data (dict): A dictionary containing chain information, with the following structure:
            {
                chain_id: {
                    'residues': {
                        (het_flag, resseq, icode): {
                            'name': str,
                            'atoms': {
                                atom_name: {
                                    'coord': np.array,
                                    'element': str
                                }
                            }
                        }
                    },
                    'sequence': str,
                    'is_het': bool
                }
            }
        structure_id (str, optional): An identifier for the structure. Defaults to "network_output".

    Returns:
        Structure: A custom Structure object containing the constructed Bio.PDB Structure.
    """
    # Create the main structure and model
    structure = BioStructure(structure_id)
    model = Model(0)
    structure.add(model)

    # Iterate through chains in the input data
    for chain_id, chain_info in chain_data.items():
        chain = Chain(chain_id)
        model.add(chain)

        # Iterate through residues in the chain
        for residue_id, residue_info in chain_info['residues'].items():
            het_flag, resseq, icode = residue_id
            residue = Residue((' ', resseq, icode), residue_info['name'], het_flag)
            chain.add(residue)

            # Iterate through atoms in the residue
            for atom_name, atom_info in residue_info['atoms'].items():
                coord = atom_info['coord']
                element = atom_info['element']
                atom = Atom(atom_name, coord, 0.0, 1.0, ' ', atom_name, None, element)
                residue.add(atom)

        # Add sequence and is_het information to the chain
        chain.sequence = chain_info['sequence']
        chain.is_het = chain_info['is_het']

    # Wrap the Bio.PDB Structure in a custom Structure object and return
    return Structure(structure[0])
