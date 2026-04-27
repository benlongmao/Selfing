"""
ChemTool — SMILES validation, descriptors, similarity, and substructure checks.

Created 2026-01-18. Requires rdkit-pypi.
"""

import logging
from typing import Dict, List, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)


class ChemTool:
    """RDKit-backed cheminformatics helper."""

    def __init__(self):
        self.name = "chem_tool"
        logger.info("ChemTool initialized")

    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """Validate SMILES and return canonical form plus atom counts."""
        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                return {
                    "valid": False,
                    "error": "Invalid SMILES syntax",
                    "smiles": smiles,
                }

            canonical_smiles = Chem.MolToSmiles(mol)

            return {
                "valid": True,
                "original_smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
            }

        except Exception as e:
            logger.error(f"Error validating SMILES {smiles}: {e}")
            return {
                "valid": False,
                "error": str(e),
                "smiles": smiles,
            }

    def calculate_properties(self, smiles: str) -> Dict[str, Any]:
        """Compute common 1D physicochemical descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                return {
                    "error": "Invalid SMILES",
                    "smiles": smiles,
                }

            properties = {
                "smiles": smiles,
                "canonical_smiles": Chem.MolToSmiles(mol),
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "hba": Descriptors.NumHAcceptors(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "tpsa": round(Descriptors.TPSA(mol), 2),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "fraction_csp3": round(Descriptors.FractionCSP3(mol), 3),
            }

            properties["lipinski"] = {
                "mw_ok": properties["molecular_weight"] <= 500,
                "logp_ok": properties["logp"] <= 5,
                "hbd_ok": properties["hbd"] <= 5,
                "hba_ok": properties["hba"] <= 10,
                "violations": Lipinski.NumHeteroatoms(mol) > 0,
            }

            return properties

        except Exception as e:
            logger.error(f"Error calculating properties for {smiles}: {e}")
            return {
                "error": str(e),
                "smiles": smiles,
            }

    def batch_validate(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Validate many SMILES strings in one call."""
        results = []
        valid_count = 0
        invalid_count = 0

        for smiles in smiles_list:
            result = self.validate_smiles(smiles)
            results.append(result)
            if result["valid"]:
                valid_count += 1
            else:
                invalid_count += 1

        return {
            "total": len(smiles_list),
            "valid": valid_count,
            "invalid": invalid_count,
            "success_rate": round(valid_count / len(smiles_list) * 100, 1) if smiles_list else 0,
            "results": results,
        }

    def compare_molecules(self, smiles1: str, smiles2: str) -> Dict[str, Any]:
        """Tanimoto similarity on Morgan fingerprints (radius 2)."""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return {
                    "error": "One or both SMILES are invalid",
                    "smiles1": smiles1,
                    "smiles2": smiles2,
                }

            fp1 = AllChem.GetMorganFingerprint(mol1, 2)
            fp2 = AllChem.GetMorganFingerprint(mol2, 2)

            from rdkit import DataStructs
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            return {
                "smiles1": smiles1,
                "smiles2": smiles2,
                "tanimoto_similarity": round(similarity, 3),
                "interpretation": self._interpret_similarity(similarity),
            }

        except Exception as e:
            logger.error(f"Error comparing molecules: {e}")
            return {
                "error": str(e),
                "smiles1": smiles1,
                "smiles2": smiles2,
            }

    def _interpret_similarity(self, similarity: float) -> str:
        """Narrative bucket for Tanimoto similarity."""
        if similarity >= 0.85:
            return "very similar"
        if similarity >= 0.70:
            return "similar"
        if similarity >= 0.50:
            return "moderately similar"
        if similarity >= 0.30:
            return "weakly similar"
        return "not similar"

    def substructure_search(self, smiles: str, substructure_smarts: str) -> Dict[str, Any]:
        """Check whether ``smiles`` contains the SMARTS pattern."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            pattern = Chem.MolFromSmarts(substructure_smarts)

            if mol is None:
                return {"error": "Invalid SMILES", "smiles": smiles}
            if pattern is None:
                return {"error": "Invalid SMARTS pattern", "smarts": substructure_smarts}

            has_match = mol.HasSubstructMatch(pattern)
            matches = mol.GetSubstructMatches(pattern)

            return {
                "smiles": smiles,
                "substructure": substructure_smarts,
                "has_match": has_match,
                "num_matches": len(matches),
                "match_atom_indices": [list(match) for match in matches] if matches else [],
            }

        except Exception as e:
            logger.error(f"Error in substructure search: {e}")
            return {
                "error": str(e),
                "smiles": smiles,
                "smarts": substructure_smarts,
            }

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Capability strings for introspection / docs."""
        return {
            "validation": [
                "validate_smiles — check SMILES syntax",
                "batch_validate — validate a list of SMILES",
            ],
            "properties": [
                "calculate_properties — MW, LogP, HBA, HBD, TPSA, rotatable bonds, aromatics, Lipinski-style flags",
            ],
            "comparison": [
                "compare_molecules — Morgan FP Tanimoto similarity",
                "substructure_search — SMARTS substructure match",
            ],
            "limitations": [
                "No 3D conformer optimization (too slow for this tool)",
                "No protein docking (needs structures and specialized engines)",
                "No quantum chemistry (needs external packages)",
            ],
        }


chem_tool = ChemTool()


def validate_smiles(smiles: str) -> Dict[str, Any]:
    """Wrapper for tool routing."""
    return chem_tool.validate_smiles(smiles)


def calculate_properties(smiles: str) -> Dict[str, Any]:
    """Wrapper for tool routing."""
    return chem_tool.calculate_properties(smiles)


def batch_validate(smiles_list: List[str]) -> Dict[str, Any]:
    """Wrapper for tool routing."""
    return chem_tool.batch_validate(smiles_list)


if __name__ == "__main__":
    print("=" * 60)
    print("ChemTool self-test")
    print("=" * 60)

    tool = ChemTool()

    print("\n[Test 1: SMILES validation]")
    result = tool.validate_smiles("c1ccccc1")
    print(f"Benzene: {result}")

    print("\n[Test 2: Properties]")
    props = tool.calculate_properties("Cc1ccccc1")
    print(f"Toluene: {props}")

    print("\n[Test 3: Batch validation]")
    smiles_list = ["c1ccccc1", "CCO", "invalid_smiles"]
    batch_result = tool.batch_validate(smiles_list)
    print(f"Success rate: {batch_result['success_rate']}%")

    print("\n" + "=" * 60)
    print("ChemTool self-test done.")
