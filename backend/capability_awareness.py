#!/usr/bin/env python3
"""
Capability boundary awareness.

Helps the agent distinguish what it can actually run (tools, local binaries) from what it
must only describe or estimate—so **predictions** are not presented as **measured outputs**.
"""
import shutil
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CapabilityAwareness:
    """Lightweight capability catalog + substring checks for prompts and guardrails."""

    def __init__(self):
        # Capability catalog (English-first for prompts); optional _match_aux adds CN/alias triggers.
        self.capabilities = {
            "can_do": {
                "python_execution": {
                    "description": "Run Python code in a sandbox",
                    "limitations": ["Sandboxed", "30s timeout", "No network in default sandbox"],
                },
                "file_operations": {
                    "description": "Read/write files under the workspace",
                    "limitations": ["Paths must stay inside the allowed workspace tree"],
                },
                "web_search": {
                    "description": "Search the web for information",
                    "limitations": ["Requires Tavily (or configured) search API"],
                },
                "text_generation": {
                    "description": "Generate text, docs, and code drafts",
                    "limitations": ["LLM output may hallucinate—verify with tools when needed"],
                },
                "data_analysis": {
                    "description": "Analyze tabular data with numpy/pandas-style Python",
                    "limitations": ["Pure in-process Python—no external HPC by default"],
                },
                "game_development": {
                    "description": "Prototype games with pygame when a display is available",
                    "limitations": ["Needs a graphical environment"],
                },
                "chemistry_calculations": {
                    "description": "Cheminformatics with RDKit (when installed)",
                    "capabilities": [
                        "Validate SMILES syntax",
                        "Compute molecular properties (MW, LogP, HBA, HBD, TPSA, …)",
                        "Batch-check SMILES lists",
                        "Tanimoto similarity",
                        "Substructure search",
                    ],
                    "limitations": [
                        "No heavy 3D conformer optimization by default",
                        "No protein docking without extra tooling",
                        "No full quantum-chemistry packages unless installed",
                        "No large-scale generative-model training (GPU/data)",
                    ],
                    "usage": "Call tools or import rdkit inside the code executor when enabled",
                },
            },
            "cannot_do": {
                "dft_calculation": {
                    "description": "Density-functional theory (DFT) quantum chemistry runs",
                    "reason": "No Gaussian/ORCA/VASP/Q-Chem… binary detected in PATH",
                    "what_i_can_do_instead": "Literature-grounded estimates and qualitative reasoning",
                },
                "molecular_dynamics": {
                    "description": "All-atom molecular dynamics (MD) simulations",
                    "reason": "No GROMACS/NAMD/AMBER… driver detected in PATH",
                    "what_i_can_do_instead": "Explain MD setup and expected outcomes at a textbook level",
                },
                "quantum_computing": {
                    "description": "Hardware quantum computing or vendor simulators you do not host",
                    "reason": "No quantum device or licensed simulator wired here",
                    "what_i_can_do_instead": "Teach concepts and toy models in prose or small classical demos",
                },
                "real_experiments": {
                    "description": "Hands-on wet-lab or bench experiments",
                    "reason": "I am software—no physical lab presence",
                    "what_i_can_do_instead": "Draft protocols and predict qualitative outcomes",
                },
                "access_external_databases": {
                    "description": "Live API pulls from PDB, ChEMBL, etc.",
                    "reason": "Sandbox network limits / no baked-in DB credentials",
                    "what_i_can_do_instead": "Describe schemas and typical fields from prior knowledge",
                },
                "train_deep_learning_models": {
                    "description": "Train large molecular generative models (VAE/GAN/Transformer stacks)",
                    "reason": "Needs GPU, data, and long unattended jobs",
                    "what_i_can_do_instead": "Rule-based RDKit workflows and sanity checks",
                },
                "molecular_docking": {
                    "description": "Protein–ligand docking (AutoDock/Vina-class pipelines)",
                    "reason": "Docking engines + receptor files not guaranteed here",
                    "what_i_can_do_instead": "Compute ligand properties and discuss pose hypotheses qualitatively",
                },
            },
        }

        # Extra substring triggers (English + CJK) beyond tokenized descriptions
        self._match_aux: Dict[str, List[str]] = {
            "python_execution": [
                "python", "run code", "code runner", "execute_python", "代码", "执行", "脚本",
            ],
            "file_operations": [
                "file", "read_file", "write_file", "workspace", "sandbox", "文件", "读写", "沙盒",
            ],
            "web_search": ["web", "search", "tavily", "lookup online", "搜索", "联网"],
            "text_generation": ["text", "write", "document", "draft", "生成", "文档", "写作"],
            "data_analysis": ["numpy", "pandas", "dataframe", "tabular", "数据分析", "表格"],
            "game_development": ["pygame", "game", "游戏"],
            "chemistry_calculations": [
                "rdkit", "smiles", "cheminformatics", "molecule", "化学", "分子", "化合物",
            ],
            "dft_calculation": [
                "dft", "gaussian", "orca", "vasp", "qchem", "density functional", "密度泛函", "量子化学",
            ],
            "molecular_dynamics": ["md", "gromacs", "gmx", "namd", "amber", "分子动力学"],
            "quantum_computing": ["quantum", "qubit", "量子计算"],
            "real_experiments": ["wet lab", "bench", "实验室", "真实实验", "湿实验"],
            "access_external_databases": [
                "pdb", "chembl", "uniprot", "database api", "数据库", "api 访问",
            ],
            "train_deep_learning_models": [
                "train", "fine-tune", "gpu", "deep learning", "深度学习", "训练模型",
            ],
            "molecular_docking": ["docking", "vina", "autodock", "protein-ligand", "对接", "蛋白", "配体"],
        }

        self._update_actual_capabilities()

    def _update_actual_capabilities(self):
        """Promote DFT/MD entries to ``can_do`` when matching binaries appear on ``PATH``."""
        dft_software = ["gaussian", "g16", "g09", "orca", "vasp", "qchem", "nwchem"]
        for soft in dft_software:
            if shutil.which(soft):
                self.capabilities["can_do"]["dft_calculation"] = {
                    "description": f"Run DFT jobs with `{soft}` when configured",
                    "limitations": ["Requires correct input decks and licenses"],
                }
                del self.capabilities["cannot_do"]["dft_calculation"]
                logger.info(f"Found DFT software: {soft}")
                break

        md_software = ["gmx", "gromacs", "namd", "amber"]
        for soft in md_software:
            if shutil.which(soft):
                self.capabilities["can_do"]["molecular_dynamics"] = {
                    "description": f"Run MD with `{soft}` when configured",
                    "limitations": ["Requires correct topology/parameterization"],
                }
                del self.capabilities["cannot_do"]["molecular_dynamics"]
                logger.info(f"Found MD software: {soft}")
                break

    def _task_match_blob(self, task: str) -> str:
        t = task or ""
        return f"{t}\n{t.lower()}"

    def _collect_keywords(self, key: str, info: Dict) -> List[str]:
        parts = key.replace("_", " ").lower().split()
        parts += str(info.get("description", "")).lower().split()
        parts += [str(x).lower() for x in info.get("capabilities", []) if isinstance(x, str)]
        parts += [str(x) for x in self._match_aux.get(key, [])]
        # de-dup while keeping order
        seen = set()
        out: List[str] = []
        for p in parts:
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def check_capability(self, task: str) -> Dict:
        """
        Heuristic match of a free-text task against ``cannot_do`` then ``can_do``.

        Args:
            task: Natural-language task line (any language; CJK substrings are matched literally)

        Returns:
            Dict with ``can_do`` (bool or None), ``confidence``, ``explanation``, plus
            ``alternative`` / ``limitations`` / ``suggestion`` when applicable.
        """
        blob = self._task_match_blob(task)

        for key, info in self.capabilities["cannot_do"].items():
            keywords = self._collect_keywords(key, info)
            if any(kw in blob for kw in keywords):
                return {
                    "can_do": False,
                    "confidence": 0.95,
                    "explanation": (
                        f"I cannot {info['description']}. Reason: {info['reason']}"
                    ),
                    "alternative": info.get("what_i_can_do_instead", ""),
                }

        for key, info in self.capabilities["can_do"].items():
            keywords = self._collect_keywords(key, info)
            if any(kw in blob for kw in keywords):
                return {
                    "can_do": True,
                    "confidence": 0.9,
                    "explanation": f"I can {info['description']}",
                    "limitations": info.get("limitations", []),
                }

        return {
            "can_do": None,
            "confidence": 0.5,
            "explanation": "I am unsure whether I can do this from the short task text alone.",
            "suggestion": "Let me try with tools, or add a few concrete constraints (software, data, deadline).",
        }

    def generate_capability_prompt(self) -> str:
        """
        Short system block listing default can/cannot boundaries for prompt injection.
        """
        core_can = [
            "Python (sandbox / ~30s)",
            "Workspace file read/write",
            "Web search (when API configured)",
            "RDKit cheminformatics (when installed)",
        ]
        core_cannot = [
            "DFT quantum chemistry (unless local engines appear on PATH)",
            "Full MD engines (unless installed)",
            "Docking pipelines (unless installed)",
            "Physical wet-lab work",
        ]

        return f"""
[Capability boundary]
✅ Can: {' | '.join(core_can)}
❌ Cannot (by default): {' | '.join(core_cannot)}
⚠️ Separate **prediction** from **computed results**; say "I am not sure" when unsure.

[Python script helpers]
Inside execute_python (when exposed): list_scripts() / run_script('script.py', ['args'])
"""

    def detect_result_claim(self, user_input: str) -> Optional[str]:
        """
        If the user asks for concrete numeric/simulation outcomes, return a caution block.

        Returns:
            Reminder string for the system prompt, or ``None``.
        """
        result_keywords = [
            "计算结果",
            "计算出",
            "算出来",
            "模拟结果",
            "DFT结果",
            "MD结果",
            "实验结果",
            "你的结果",
            "你算的",
            "你计算的",
            "数值是多少",
            "频率是多少",
            "能量是多少",
            "单点能",
            "优化结构",
            "HOMO",
            "LUMO",
            "binding energy",
            "final energy",
            "single-point",
            "computed result",
            "simulation result",
            "your numbers",
            "what is the value",
            "what frequency",
            "what energy",
            "show me the numbers",
            "what did you compute",
        ]

        user_text = user_input or ""

        for keyword in result_keywords:
            if keyword in user_text:
                return f"""
[⚠️ Result-claim check]
The user is asking about "{keyword}".

Before answering, confirm:
1. Did I actually run that computation here?
2. Where do the numbers come from? (real tool stdout / a reasoned estimate / literature)
3. If it is a prediction, say explicitly that it is a **theoretical estimate**, not a lab or DFT output.
"""

        return None


_capability_awareness: Optional[CapabilityAwareness] = None


def get_capability_awareness() -> CapabilityAwareness:
    """Lazy singleton for ``CapabilityAwareness``."""
    global _capability_awareness
    if _capability_awareness is None:
        _capability_awareness = CapabilityAwareness()
    return _capability_awareness
