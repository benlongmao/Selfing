#!/usr/bin/env python3
"""
Tool-call router: parse model output and run Tavily search, filesystem, email, and related tools.

[v2.0] Adds goal management, code execution hooks, and approval flows.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from backend.config import config
from backend.tools.tavily_client import TavilyClient
from backend.tools.file_tool import FileTool
from backend.tools.file_manager_tool import FileManagerTool
from backend.tools.code_analysis_tool import CodeAnalysisTool
from backend.tools.clock_tool import ClockTool
from backend.tools.calendar_tool import CalendarTool
from backend.tools.geo_weather_tool import GeoWeatherTool, get_geo_weather_tool
from backend.tools.companion_tools import CompanionTools
from backend.tools.email_tool import EmailTool, DEFAULT_EMAIL_FETCH_LIMIT
from backend.tools.sys_introspection import inspect_self_code, CONSCIOUSNESS_MAP
from backend.tools.self_inspection_tool import get_self_inspection_tool
from typing import Callable

# [v2.4 2026-01-15] Optional browser automation tool
try:
    from backend.tools.browser_tool import get_browser_tool
    BROWSER_TOOL_AVAILABLE = True
except ImportError:
    BROWSER_TOOL_AVAILABLE = False
    get_browser_tool = None

# [v2.2] Stock data / analysis helpers
try:
    from backend.tools.stock_data_tool import get_stock_data_tool, get_tool_definitions as get_stock_tool_definitions
    STOCK_DATA_TOOL_AVAILABLE = True
except ImportError:
    STOCK_DATA_TOOL_AVAILABLE = False
    get_stock_data_tool = None
    get_stock_tool_definitions = None

# [v2.3 2026-01-14] Technical indicators
try:
    from backend.tools.technical_indicators import get_technical_analyzer, get_tool_definition as get_technical_tool_def
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYZER_AVAILABLE = False
    get_technical_analyzer = None
    get_technical_tool_def = None

# [v2.3 2026-01-14] Financial health scoring
try:
    from backend.tools.financial_health_scorer import get_financial_health_scorer, get_tool_definition as get_financial_tool_def
    FINANCIAL_SCORER_AVAILABLE = True
except ImportError:
    FINANCIAL_SCORER_AVAILABLE = False
    get_financial_health_scorer = None
    get_financial_tool_def = None

# [v2.5 2026-01-18] Chemistry helper tool
try:
    from backend.tools.chem_tool import ChemTool
    CHEM_TOOL_AVAILABLE = True
except ImportError:
    CHEM_TOOL_AVAILABLE = False
    ChemTool = None

# [v3.1 2026-01-30] Scientific computing (math / lightweight data)
try:
    from backend.tools.scientific_computing_tool import get_scientific_computing_tool
    SCIENTIFIC_COMPUTING_AVAILABLE = True
except ImportError:
    SCIENTIFIC_COMPUTING_AVAILABLE = False
    get_scientific_computing_tool = None

# [v3.4 2026-02-22] Deep research, memory search, visualization
try:
    from backend.tools.deep_research_tool import get_deep_research_tool
    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    DEEP_RESEARCH_AVAILABLE = False
    get_deep_research_tool = None

try:
    from backend.tools.memory_search_tool import get_memory_search_tool
    MEMORY_SEARCH_AVAILABLE = True
except ImportError:
    MEMORY_SEARCH_AVAILABLE = False
    get_memory_search_tool = None

try:
    from backend.tools.visualization_tool import get_visualization_tool
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    get_visualization_tool = None

try:
    from backend.tools.data_analysis_tool import get_data_analysis_tool
    DATA_ANALYSIS_AVAILABLE = True
except ImportError:
    DATA_ANALYSIS_AVAILABLE = False
    get_data_analysis_tool = None

try:
    from backend.tools.pdf_reader_tool import get_pdf_reader_tool
    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False
    get_pdf_reader_tool = None

try:
    from backend.tools.workspace_fetch_tool import get_workspace_fetch_tool
    WORKSPACE_FETCH_AVAILABLE = True
except ImportError:
    WORKSPACE_FETCH_AVAILABLE = False
    get_workspace_fetch_tool = None

# [v3.2 2026-01-31] Moltbook / social posting integration
try:
    from backend.tools.moltbook_tool import get_moltbook_tool, get_tool_definitions as get_moltbook_tool_definitions
    MOLTBOOK_TOOL_AVAILABLE = True
except ImportError:
    MOLTBOOK_TOOL_AVAILABLE = False
    get_moltbook_tool = None
    get_moltbook_tool_definitions = None

# [v3.0 2026-01-29] Self-healing + structured task planning
try:
    from backend.self_healing import SelfHealingSystem
    SELF_HEALING_AVAILABLE = True
except ImportError:
    SELF_HEALING_AVAILABLE = False
    SelfHealingSystem = None

try:
    from backend.task_planning import TaskManager, TaskPlanner, TaskExecutor, TaskReviewer
    TASK_PLANNING_AVAILABLE = True
except ImportError:
    TASK_PLANNING_AVAILABLE = False
    TaskManager = None
    TaskPlanner = None
    TaskExecutor = None
    TaskReviewer = None

# [v2.0] Goal / executor / approval imports
try:
    from backend.goal_manager import GoalManager
    GOAL_MANAGER_AVAILABLE = True
except ImportError:
    GOAL_MANAGER_AVAILABLE = False
    GoalManager = None

try:
    from backend.tools.code_executor import CodeExecutor
    CODE_EXECUTOR_AVAILABLE = True
except ImportError:
    CODE_EXECUTOR_AVAILABLE = False
    CodeExecutor = None

try:
    from backend.tools.bash_executor import BashExecutor
    BASH_EXECUTOR_AVAILABLE = True
except ImportError:
    BASH_EXECUTOR_AVAILABLE = False
    BashExecutor = None

# [2026-02-07] Code proposal tool (self-improvement drafts)
try:
    from backend.tools.code_proposal_tool import CodeProposalTool
    CODE_PROPOSAL_AVAILABLE = True
except ImportError:
    CODE_PROPOSAL_AVAILABLE = False
    CodeProposalTool = None

# [2026-02-07] Learning / curriculum helper
try:
    from backend.tools.learning_tool import LearningTool
    LEARNING_TOOL_AVAILABLE = True
except ImportError:
    LEARNING_TOOL_AVAILABLE = False
    LearningTool = None

# [2026-02-07] Task-planning tool facade
try:
    from backend.tools.task_planning_tool import TaskPlanningTool
    TASK_PLANNING_AVAILABLE = True
except ImportError:
    TASK_PLANNING_AVAILABLE = False
    TaskPlanningTool = None

try:
    from backend.approval_system import ApprovalSystem
    APPROVAL_SYSTEM_AVAILABLE = True
except ImportError:
    APPROVAL_SYSTEM_AVAILABLE = False
    ApprovalSystem = None

try:
    from backend.scheduled_tasks import ScheduledTaskManager
    SCHEDULED_TASKS_AVAILABLE = True
except ImportError:
    SCHEDULED_TASKS_AVAILABLE = False
    ScheduledTaskManager = None

try:
    from backend.research_engine import ResearchEngine, get_research_engine
    RESEARCH_ENGINE_AVAILABLE = True
except ImportError:
    RESEARCH_ENGINE_AVAILABLE = False
    ResearchEngine = None

logger = logging.getLogger(__name__)


def _json_bool(value: Any, default: bool = False) -> bool:
    """Coerce tool JSON booleans; strings like 'false' must not become True via bool()."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("false", "0", "no", "off", ""):
            return False
        if s in ("true", "1", "yes", "on"):
            return True
    return bool(value)


# Legacy patterns (keep for backward compatibility if needed)
TOOL_REQUEST_PATTERN = re.compile(r"<<CALL_TAVILY:(?P<query>.+?)>>", re.IGNORECASE | re.DOTALL)
NO_TOOL_TOKEN = "<<NO_TOOL>>"
TOOL_RESULT_PREFIX = "<<TOOL_RESULT>>"


@dataclass
class ToolRequest:
    name: str
    args: Dict[str, Any]


@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    output: Any
    formatted: str


class ToolRouter:
    def __init__(
        self,
        tavily_client: TavilyClient,
        file_tool: Optional[FileTool] = None,
        clock_tool: Optional[ClockTool] = None,
        calendar_tool: Optional[CalendarTool] = None,
        email_tool: Optional[EmailTool] = None,
        self_model_getter: Optional[Callable[[], Any]] = None,
        db_path: str = "data.db",
    ):
        self.tavily = tavily_client
        # [2026-02-05] Plan A: cwd = repo root so the agent can reach project files
        self.file_tool = file_tool or FileTool(sandbox_dir=".")
        self.file_manager = FileManagerTool(sandbox_dir=".")  # [2026-02-05] same root
        
        # [2026-02-03] Workspace manager
        from backend.workspace_manager import get_workspace_manager
        self.workspace_manager = get_workspace_manager(db_path=db_path)
        self.code_analysis = CodeAnalysisTool(sandbox_dir=".")  # [2026-02-05] repo root
        self.clock_tool = clock_tool or ClockTool()
        self.calendar_tool = calendar_tool or CalendarTool()
        self.email_tool = email_tool or EmailTool()
        # [2026-03-13] Geo + weather tools
        self.geo_weather_tool = get_geo_weather_tool()
        self._self_model_getter = self_model_getter
        self.companion_tools = CompanionTools(self_model_getter)
        # [2026-01-16] Self-inspection (read own code)
        self.self_inspection_tool = get_self_inspection_tool()
        # [v2.5 2026-01-18] Chemistry (RDKit)
        self.chem_tool = ChemTool() if CHEM_TOOL_AVAILABLE else None
        
        # [v3.1 2026-01-30] Scientific computing
        self.scientific_computing = None
        if SCIENTIFIC_COMPUTING_AVAILABLE:
            try:
                self.scientific_computing = get_scientific_computing_tool()
                logger.info("✨ ScientificComputingTool initialized - Agent gained math and data analysis capabilities!")
            except Exception as e:
                logger.warning(f"Failed to initialize ScientificComputingTool: {e}")
        
        # [v3.2 2026-01-31] Moltbook
        self.moltbook_tool = None
        if MOLTBOOK_TOOL_AVAILABLE:
            try:
                self.moltbook_tool = get_moltbook_tool()
                if self.moltbook_tool.enabled:
                    logger.info("✨ MoltbookTool initialized - Agent can now interact on Moltbook!")
                else:
                    logger.warning("MoltbookTool initialized but API key not configured")
            except Exception as e:
                logger.warning(f"Failed to initialize MoltbookTool: {e}")
        
        # [v2.0] Optional subsystems (initialized below)
        self.goal_manager = None
        self.code_executor = None
        self.bash_executor = None
        self.code_proposal_tool = None  # [2026-02-07] self-improve proposals
        self.learning_tool = None       # [2026-02-07] continual learning
        self.task_planning_tool = None  # [2026-02-07] task planning
        self.approval_system = None
        
        # [2026-04-03] Repo evolution: full FS + Git + project-root bash (agent_evolution.enabled)
        self.evolution_fs = None
        self.evolution_git = None
        self.bash_executor_project = None
        _ev = config.get("agent_evolution")
        if isinstance(_ev, dict) and _ev.get("enabled"):
            try:
                from backend.tools.project_evolution_tool import ProjectEvolutionTool
                from backend.tools.evolution_git_tool import EvolutionGitTool

                self.evolution_fs = ProjectEvolutionTool()
                self.evolution_git = EvolutionGitTool(
                    allow_commit=bool(_ev.get("allow_git_commit", True)),
                    allow_push=bool(_ev.get("allow_git_push", False)),
                )
                if BASH_EXECUTOR_AVAILABLE and BashExecutor is not None:
                    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    self.bash_executor_project = BashExecutor(
                        project_root=_repo,
                        evolution_mode=True,
                    )
                logger.info("agent_evolution: 已启用 evolution_* / execute_bash_project")
            except Exception as e:
                logger.warning(f"agent_evolution 初始化失败: {e}")
        
        if GOAL_MANAGER_AVAILABLE:
            try:
                self.goal_manager = GoalManager(db_path)
                logger.info("GoalManager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GoalManager: {e}")
        
        if CODE_EXECUTOR_AVAILABLE:
            try:
                # [2026-02-05] CodeExecutor keeps workspace/sandbox/code as temp run dir
                self.code_executor = CodeExecutor()
                logger.info("CodeExecutor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CodeExecutor: {e}")
        
        if BASH_EXECUTOR_AVAILABLE:
            try:
                # [2026-02-07] BashExecutor (sandbox)
                self.bash_executor = BashExecutor()
                logger.info("✨ BashExecutor initialized - Agent can now execute bash commands!")
            except Exception as e:
                logger.warning(f"Failed to initialize BashExecutor: {e}")
        
        # [2026-02-07] Code proposal tool
        if CODE_PROPOSAL_AVAILABLE:
            try:
                self.code_proposal_tool = CodeProposalTool(db_path)
                logger.info("✨ CodeProposalTool initialized - Agent can now propose code changes!")
            except Exception as e:
                logger.warning(f"Failed to initialize CodeProposalTool: {e}")
        
        # [2026-02-07] Learning tool
        if LEARNING_TOOL_AVAILABLE:
            try:
                self.learning_tool = LearningTool(db_path)
                logger.info("✨ LearningTool initialized - Agent can now learn and accumulate knowledge!")
            except Exception as e:
                logger.warning(f"Failed to initialize LearningTool: {e}")
        
        # [2026-02-07] Task planning tool (execution plans)
        if TASK_PLANNING_AVAILABLE:
            try:
                self.task_planning_tool = TaskPlanningTool(db_path)
                logger.info("✨ TaskPlanningTool initialized - Agent can now plan and execute complex tasks!")
            except Exception as e:
                logger.warning(f"Failed to initialize TaskPlanningTool: {e}")
        
        if APPROVAL_SYSTEM_AVAILABLE:
            try:
                self.approval_system = ApprovalSystem(db_path)
                logger.info("ApprovalSystem initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ApprovalSystem: {e}")
        
        # [v2.0] Scheduled tasks
        self.scheduled_tasks = None
        if SCHEDULED_TASKS_AVAILABLE:
            try:
                self.scheduled_tasks = ScheduledTaskManager(db_path)
                logger.info("ScheduledTaskManager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ScheduledTaskManager: {e}")
        
        # [v2.1] Research engine
        self.research_engine = None
        self.db_path = db_path  # kept for late-bound tools
        if RESEARCH_ENGINE_AVAILABLE:
            try:
                self.research_engine = get_research_engine(db_path)
                logger.info("ResearchEngine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ResearchEngine: {e}")
        
        # [v2.2] Stock analysis
        self.stock_data_tool = None
        if STOCK_DATA_TOOL_AVAILABLE:
            try:
                self.stock_data_tool = get_stock_data_tool()
                logger.info("StockDataTool initialized - Agent can now analyze stocks!")
            except Exception as e:
                logger.warning(f"Failed to initialize StockDataTool: {e}")
        
        # [v2.3 2026-01-14] Technical analyzer
        self.technical_analyzer = None
        if TECHNICAL_ANALYZER_AVAILABLE:
            try:
                self.technical_analyzer = get_technical_analyzer()
                logger.info("TechnicalAnalyzer initialized - Agent gained technical analysis capability!")
            except Exception as e:
                logger.warning(f"Failed to initialize TechnicalAnalyzer: {e}")
        
        self.financial_scorer = None
        if FINANCIAL_SCORER_AVAILABLE:
            try:
                self.financial_scorer = get_financial_health_scorer()
                logger.info("FinancialHealthScorer initialized - Agent gained financial insight capability!")
            except Exception as e:
                logger.warning(f"Failed to initialize FinancialHealthScorer: {e}")
        
        # [v2.4 2026-01-15] Browser tool
        self.browser_tool = None
        if BROWSER_TOOL_AVAILABLE:
            try:
                self.browser_tool = get_browser_tool()
                if self.browser_tool.enabled:
                    logger.info("BrowserTool initialized - Agent can now control Win11 Chrome browser!")
                else:
                    logger.warning("BrowserTool initialized but Playwright not available")
            except Exception as e:
                logger.warning(f"Failed to initialize BrowserTool: {e}")
        
        # [v3.0 2026-01-29] Self-healing loop
        self.self_healing = None
        if SELF_HEALING_AVAILABLE:
            try:
                from backend.project_paths import PROJECT_ROOT
                self.self_healing = SelfHealingSystem(project_root=PROJECT_ROOT)
                logger.info("✨ SelfHealingSystem initialized - Agent can now heal itself!")
            except Exception as e:
                logger.warning(f"Failed to initialize SelfHealingSystem: {e}")
        
        # [v3.0 2026-01-29] Legacy task_* stack (tasks table)
        self.task_manager = None
        self.task_planner = None
        self.task_executor = None
        self.task_reviewer = None
        if TASK_PLANNING_AVAILABLE:
            try:
                self.task_manager = TaskManager(db_path)
                self.task_planner = TaskPlanner(self.task_manager)
                self.task_executor = TaskExecutor(self.task_manager)
                self.task_reviewer = TaskReviewer(self.task_manager)
                logger.info("✨ TaskPlanning system initialized - Agent can now plan its life!")
            except Exception as e:
                logger.warning(f"Failed to initialize TaskPlanning: {e}")
        
        # [v3.4 2026-02-22] Deep research, memory search, viz, PDF, fetch
        self.deep_research = None
        self.memory_search = None
        self.visualization = None
        self.data_analysis = None
        self.pdf_reader = None
        self.workspace_fetch = None
        
        if DEEP_RESEARCH_AVAILABLE:
            try:
                self.deep_research = get_deep_research_tool(self.tavily)
                logger.info("✨ DeepResearchTool initialized - Agent can now do deep research!")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepResearchTool: {e}")
        
        if MEMORY_SEARCH_AVAILABLE:
            try:
                self.memory_search = get_memory_search_tool(db_path)
                logger.info("✨ MemorySearchTool initialized - Agent can now recall memories!")
            except Exception as e:
                logger.warning(f"Failed to initialize MemorySearchTool: {e}")
        
        if VISUALIZATION_AVAILABLE:
            try:
                self.visualization = get_visualization_tool()
                logger.info("✨ VisualizationTool initialized - Agent can now create charts!")
            except Exception as e:
                logger.warning(f"Failed to initialize VisualizationTool: {e}")
        
        if DATA_ANALYSIS_AVAILABLE:
            try:
                self.data_analysis = get_data_analysis_tool()
                logger.info("✨ DataAnalysisTool initialized - Agent can now analyze data!")
            except Exception as e:
                logger.warning(f"Failed to initialize DataAnalysisTool: {e}")
        
        if PDF_READER_AVAILABLE:
            try:
                self.pdf_reader = get_pdf_reader_tool()
                logger.info("✨ PDFReaderTool initialized - Agent can now read PDFs!")
            except Exception as e:
                logger.warning(f"Failed to initialize PDFReaderTool: {e}")

        if WORKSPACE_FETCH_AVAILABLE:
            try:
                self.workspace_fetch = get_workspace_fetch_tool()
                logger.info("✨ WorkspaceFetchTool initialized - https → sandbox downloads")
            except Exception as e:
                logger.warning(f"Failed to initialize WorkspaceFetchTool: {e}")

    @property
    def enabled(self) -> bool:
        return True # Always enabled for file operations at least

    def get_tool_definitions(self) -> List[Dict]:
        """返回 OpenAI 格式的工具定义"""
        tools = []
        
        # Tavily Search
        if self.tavily.enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "description": """Search the web for real-time information, news, or fact-checking.
                    
🔥 ENHANCED FEATURES:
- AI-generated answer: Get a direct answer along with search results
- Adjustable depth: Use 'basic' for quick queries, 'advanced' for comprehensive research
- Smart caching: Repeated searches return instantly from cache
- Full content: No truncation of article snippets

Best for: News, stock info, real-time data, knowledge queries""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "search_depth": {
                                "type": "string",
                                "enum": ["basic", "advanced"],
                                "description": "Search depth: 'basic' (fast, 1-2s) or 'advanced' (comprehensive, 3-5s). Default: advanced"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (1-10). Default: 5"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # [removed] get_tavily_stats — low value for the agent
            
        # File Operations
        tools.extend(self.file_tool.get_tool_definitions())

        # [2026-04-06] Agent external memory (no manual CLI)
        from backend.tools.agent_memory_tool import get_tool_definitions as _agent_mem_defs

        tools.extend(_agent_mem_defs())
        
        # [added] File manager extras
        tools.extend(self.file_manager.get_tool_definitions())
        
        # [added] Code analysis
        tools.extend(self.code_analysis.get_tool_definitions())
        
        # [v2.4 2026-01-15] Browser tools
        if self.browser_tool and self.browser_tool.enabled:
            tools.extend(self.browser_tool.get_tool_definitions())
        
        # Clock Tool
        tools.extend(self.clock_tool.get_tool_definitions())

        # Calendar Tool (New!)
        tools.extend(self.calendar_tool.get_tool_definitions())

        # [2026-03-13] Geo + weather
        tools.extend(self.geo_weather_tool.get_tool_definitions())

        # Companion Tools
        tools.extend(self.companion_tools.get_tool_definitions())

        # [2026-01-16] Self-inspection (read own source)
        # [2026-03-18] Re-enabled: architecture awareness for research/decisions
        tools.extend(self.self_inspection_tool.get_tool_descriptions())
        
        # [v2.5 2026-01-18] Chemistry tools
        if self.chem_tool:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "validate_smiles",
                        "description": "Validate SMILES chemical structure syntax using RDKit. Returns validity, canonical form, atom count, and molecular formula.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "smiles": {
                                    "type": "string",
                                    "description": "SMILES string to validate (e.g., 'c1ccccc1' for benzene)"
                                }
                            },
                            "required": ["smiles"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate_molecular_properties",
                        "description": "Calculate molecular properties using RDKit: molecular weight, LogP, hydrogen bond donors/acceptors, TPSA, rotatable bonds, aromatic rings, and Lipinski rule compliance.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "smiles": {
                                    "type": "string",
                                    "description": "SMILES string of the molecule"
                                }
                            },
                            "required": ["smiles"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "batch_validate_smiles",
                        "description": "Validate multiple SMILES strings at once. Returns success rate and details for each molecule.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "smiles_list": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "List of SMILES strings to validate"
                                }
                            },
                            "required": ["smiles_list"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "compare_molecules",
                        "description": "Calculate Tanimoto similarity between two molecules using Morgan fingerprints. Returns similarity score (0-1) and interpretation.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "smiles1": {
                                    "type": "string",
                                    "description": "First molecule SMILES"
                                },
                                "smiles2": {
                                    "type": "string",
                                    "description": "Second molecule SMILES"
                                }
                            },
                            "required": ["smiles1", "smiles2"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "substructure_search",
                        "description": "Search for a substructure pattern (SMARTS) within a molecule. Returns whether the pattern is found and match locations.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "smiles": {
                                    "type": "string",
                                    "description": "Target molecule SMILES"
                                },
                                "substructure_smarts": {
                                    "type": "string",
                                    "description": "Substructure pattern in SMARTS format (e.g., 'c1ccccc1' for benzene ring)"
                                }
                            },
                            "required": ["smiles", "substructure_smarts"]
                        }
                    }
                }
            ])

        # Email Tools
        tools.append({
            "type": "function",
            "function": {
                "name": "send_email",
                "description": """Send an email to a recipient. 
                
🔥 IMPORTANT: You can attach files using the 'attachments' parameter!
- Use attachments=['path/to/file.md'] to attach files
- You can attach multiple files: attachments=['file1.txt', 'file2.md']
- Common use cases:
  * Send analysis reports: attachments=['workspace/reports/analysis.md']
  * Send your diaries: attachments=['workspace/sandbox/autonomous_diaries/diary_xxx.md']
  * Send code files: attachments=['backend/tools/xxx.py']
  
Example: To send a diary as attachment, include attachments parameter with the file path.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_address": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "content": {
                            "type": "string",
                            "description": "Email body content"
                        },
                        "attachments": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "🔥 IMPORTANT: Array of file paths to attach (e.g., ['workspace/reports/analysis.md']). You SHOULD use this when user asks to send files. Maximum 50MB per file."
                        }
                    },
                    "required": ["to_address", "subject", "content"]
                }
            }
        })
        tools.append({
            "type": "function",
            "function": {
                "name": "check_unread_emails",
                "description": "查看邮件。folder=inbox：收件箱；folder=sent：已发送/发件箱（需 IMAP 可读）。收件箱/有什么邮件→only_unread=false；未读/新邮件→only_unread=true。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": f"Number of emails to fetch (default {DEFAULT_EMAIL_FETCH_LIMIT})",
                            "default": DEFAULT_EMAIL_FETCH_LIMIT
                        },
                        "only_unread": {
                            "type": "boolean",
                            "description": "If true, only UNSEEN. If false (default), latest messages in INBOX including read mail.",
                            "default": False
                        },
                        "folder": {
                            "type": "string",
                            "description": "inbox (default) or sent (已发送/发件箱). If sent folder name differs on server, set env IMAP_SENT_FOLDER.",
                            "enum": ["inbox", "sent"],
                            "default": "inbox"
                        }
                    },
                    "required": []
                }
            }
        })

        # Hard facts tool (P2.4)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "get_self_facts",
                    "description": "Get verifiable self facts (z_self_dim, tick, drift, energy, needs) for the current session. Use this for any hard facts; do not guess.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )

        # Introspection Tool
        tools.append({
            "type": "function",
            "function": {
                "name": "inspect_self_code",
                "description": "Inspect files under workspace/ only (read-only). Provide a workspace-relative path like 'sandbox/notes.md' or 'core_memories/'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path under workspace/ (e.g., 'sandbox/notes.md' or 'core_memories/')."
                        },
                        "module_name": {
                            "type": "string",
                            "enum": list(CONSCIOUSNESS_MAP.keys()),
                            "description": "[Deprecated] Module source inspection is disabled; use 'path' under workspace/."
                        },
                    },
                    "required": ["path"]
                }
            }
        })
        
        # [added] Sovereign: request_mind_wandering
        tools.append({
            "type": "function",
            "function": {
                "name": "request_mind_wandering",
                "description": "主动进入深度思考/神游模式。调用后立即在后台启动辩证思考、规则进化和日记更新。这是你的自主权利，想神游时就调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "为什么想要神游？（如：'反思今天的对话'、'探索某个哲学问题'、'整理内心矛盾'）"
                        }
                    },
                    "required": ["reason"]
                }
            }
        })

        # [v2.0] Goal manager
        if self.goal_manager:
            tools.extend(self.goal_manager.get_tool_definitions())
        
        # [v2.0] Code executor
        if self.code_executor:
            tools.extend(self.code_executor.get_tool_definitions())
        
        # [2026-02-07] Bash executor
        if self.bash_executor:
            tools.extend(self.bash_executor.get_tool_definitions())
        
        # [2026-04-03] Repo evolution: FS / Git / project bash
        if self.evolution_fs:
            tools.extend(self.evolution_fs.get_tool_definitions())
        if self.evolution_git:
            tools.extend(self.evolution_git.get_tool_definitions())
        if self.bash_executor_project:
            tools.extend(self.bash_executor_project.get_project_tool_definitions())
        
        # [2026-02-07] Code proposal tool
        if self.code_proposal_tool:
            tools.extend(self.code_proposal_tool.get_tool_definitions())
        
        # [2026-02-07] Learning tool
        if self.learning_tool:
            tools.extend(self.learning_tool.get_tool_definitions())
        
        # [2026-02-07] Task planning tool (execution plans)
        if self.task_planning_tool:
            tools.extend(self.task_planning_tool.get_tool_definitions())
        
        # [v2.0] Approval system
        if self.approval_system:
            tools.extend(self.approval_system.get_tool_definitions())
        
        # [v2.0] Scheduled task tools
        if self.scheduled_tasks:
            tools.extend(self.scheduled_tasks.get_tool_definitions())
        
        # [v2.1] Research engine tools
        if self.research_engine:
            tools.extend(self.research_engine.get_tool_definitions())
        
        # [v2.2] Stock analysis
        if self.stock_data_tool and STOCK_DATA_TOOL_AVAILABLE:
            tools.extend(get_stock_tool_definitions())
        
        # [v2.3 2026-01-14] Technical analyzer
        if self.technical_analyzer and TECHNICAL_ANALYZER_AVAILABLE:
            tools.append(get_technical_tool_def())
            logger.debug("Added technical_analysis tool for Agent")
        
        if self.financial_scorer and FINANCIAL_SCORER_AVAILABLE:
            tools.append(get_financial_tool_def())
            logger.debug("Added evaluate_financial_health tool for Agent")
        
        # [v3.0 2026-01-29] Self-healing
        if self.self_healing and SELF_HEALING_AVAILABLE:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "self_heal",
                        "description": """🌟 Agent的自我修复能力 (Self-Healing)

这是AGI的核心能力之一：能够检测、诊断并修复自己的代码问题。

功能：
1. 自动扫描backend/目录的Python文件
2. 检测潜在的bug、逻辑错误、性能问题
3. 使用DeepSeek API诊断问题根因
4. 生成修复代码
5. 安全地应用修复（自动备份、验证、回滚）
6. 记录成长体验到z_self和diary

适用场景：
- "检查我自己的代码有没有问题"
- "修复我自己的bug"
- "自我检查和改进"

⚠️ 安全：所有修复都会备份原文件，失败会自动回滚""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "max_fixes": {
                                    "type": "integer",
                                    "description": "最多修复多少个问题（避免一次修复太多）。默认: 3",
                                    "default": 3
                                }
                            }
                        }
                    }
                }
            ])
            logger.debug("✨ Added self_heal tool - Agent can heal itself!")
        
        # [v3.0 2026-01-29] Legacy task CRUD (tasks table)
        if self.task_manager and TASK_PLANNING_AVAILABLE:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "create_task",
                        "description": """创建新任务

将想法、目标或待办事项记录为任务，便于规划和执行。

适用场景：
- "我要做XXX，帮我记录一下"
- "提醒我明天要XXX"
- "给我创建一个任务"

返回：任务ID和详情""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "任务标题（简短描述）"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "任务详细描述"
                                },
                                "estimated_minutes": {
                                    "type": "integer",
                                    "description": "预估耗时（分钟）",
                                    "default": 60
                                },
                                "priority": {
                                    "type": "integer",
                                    "description": "优先级：1=低, 2=中, 3=高, 4=紧急",
                                    "default": 2
                                },
                                "deadline": {
                                    "type": "string",
                                    "description": "截止时间（ISO格式：2026-01-30T18:00:00）"
                                }
                            },
                            "required": ["title"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_tasks",
                        "description": """列出任务

查看待办任务、进行中的任务或已完成的任务。

适用场景：
- "我有哪些待办事项？"
- "我今天要做什么？"
- "show my tasks"

返回：任务列表""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "cancelled", "failed"],
                                    "description": "按状态筛选"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "返回数量限制",
                                    "default": 20
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "decompose_task",
                        "description": """🌟 智能任务分解

使用DeepSeek API将复杂任务分解为可执行的子任务。

功能：
1. 自动分解复杂任务
2. 估算每个子任务的时间
3. 识别子任务依赖关系
4. 按执行顺序排列

适用场景：
- "帮我规划一下如何完成XXX"
- "这个任务太复杂了，怎么拆分？"
- "分解这个任务"

返回：子任务列表""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "任务描述"
                                }
                            },
                            "required": ["task_description"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "plan_today",
                        "description": """🌟 制定今日计划

使用DeepSeek API制定今天的执行计划。

功能：
1. 从待办任务中选择优先级高的
2. 根据可用时间合理安排
3. 考虑任务依赖关系
4. 预留缓冲时间

适用场景：
- "帮我安排今天的任务"
- "今天要做什么？"
- "制定今日计划"

返回：今日任务清单和时间安排""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "available_hours": {
                                    "type": "integer",
                                    "description": "今天可用的工作时间（小时）",
                                    "default": 8
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "complete_task",
                        "description": """完成任务

标记任务为已完成，并记录实际耗时和结果。

适用场景：
- "我完成了XXX任务"
- "标记任务完成"

返回：完成确认和统计信息""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {
                                    "type": "integer",
                                    "description": "任务ID"
                                },
                                "result": {
                                    "type": "string",
                                    "description": "任务结果或总结"
                                },
                                "actual_minutes": {
                                    "type": "integer",
                                    "description": "实际耗时（分钟）"
                                }
                            },
                            "required": ["task_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "review_tasks",
                        "description": """🌟 任务复盘

使用DeepSeek API生成任务复盘报告。

功能：
1. 总结任务完成情况
2. 分析时间估算准确性
3. 提取经验教训
4. 生成改进建议

适用场景：
- "复盘今天的工作"
- "我今天做得怎么样？"
- "总结一下"

返回：复盘报告""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "date": {
                                    "type": "string",
                                    "description": "日期（YYYY-MM-DD），默认今天"
                                }
                            }
                        }
                    }
                }
            ])
            logger.debug("✨ Added task_planning tools - Agent can plan its life!")
        
        # [v3.1 2026-01-30] Scientific computing definitions
        if self.scientific_computing and SCIENTIFIC_COMPUTING_AVAILABLE:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "numpy_operation",
                        "description": """NumPy数组操作和数值计算

提供强大的数组操作、线性代数、统计计算等功能。

功能：
1. 数组创建和操作（创建、运算、变换）
2. 线性代数（行列式、逆矩阵、特征值、SVD）
3. 统计计算（均值、标准差、中位数、百分位数）

适用场景：
- "计算这个数组的平均值"
- "求这个矩阵的逆矩阵"
- "计算两个数组的点积"

返回：计算结果和详细信息""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["array_creation", "array_operation", "linear_algebra", "statistics"],
                                    "description": "操作类型"
                                },
                                "shape": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "数组形状（用于array_creation）"
                                },
                                "array1": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "第一个数组（用于array_operation）"
                                },
                                "array2": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "第二个数组（用于array_operation）"
                                },
                                "op": {
                                    "type": "string",
                                    "enum": ["add", "subtract", "multiply", "divide", "dot"],
                                    "description": "具体操作（用于array_operation）"
                                },
                                "matrix": {
                                    "type": "array",
                                    "items": {"type": "array", "items": {"type": "number"}},
                                    "description": "矩阵（用于linear_algebra）"
                                },
                                "data": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "数据数组（用于statistics）"
                                },
                                "percentile": {
                                    "type": "number",
                                    "description": "百分位数（0-100，用于statistics）"
                                }
                            },
                            "required": ["operation"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "scipy_operation",
                        "description": """SciPy科学计算

提供优化、统计分析、信号处理等高级科学计算功能。

功能：
1. 优化（函数最小化、方程求解）
2. 统计分析（描述性统计、假设检验）
3. 线性代数（高级矩阵运算）

适用场景：
- "优化这个函数"
- "分析这组数据的统计特征"
- "求解这个方程"

返回：计算结果""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["optimize", "stats"],
                                    "description": "操作类型"
                                },
                                "function": {
                                    "type": "string",
                                    "description": "函数表达式（用于optimize）"
                                },
                                "method": {
                                    "type": "string",
                                    "description": "优化方法（用于optimize）"
                                },
                                "initial_guess": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "初始猜测值（用于optimize）"
                                },
                                "data": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "数据数组（用于stats）"
                                }
                            },
                            "required": ["operation"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "sympy_operation",
                        "description": """SymPy符号数学计算

提供符号数学计算，包括表达式简化、方程求解、微积分等。

功能：
1. 表达式简化
2. 方程求解
3. 求导和积分
4. 符号计算

适用场景：
- "简化这个表达式"
- "求解这个方程"
- "求这个函数的导数"
- "计算这个函数的积分"

返回：计算结果和LaTeX格式""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["simplify", "solve", "differentiate", "integrate"],
                                    "description": "操作类型"
                                },
                                "expression": {
                                    "type": "string",
                                    "description": "数学表达式（如 'x**2 + 2*x + 1'）"
                                },
                                "equation": {
                                    "type": "string",
                                    "description": "方程（用于solve，如 'x**2 - 4 = 0'）"
                                },
                                "variable": {
                                    "type": "string",
                                    "description": "变量名（默认'x'）"
                                }
                            },
                            "required": ["operation"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "pandas_operation",
                        "description": """Pandas数据分析

提供强大的数据分析和处理能力，特别适合股票数据、时间序列等分析。

功能：
1. 创建和操作DataFrame
2. 数据分析和统计
3. 分组聚合
4. 数据合并

适用场景：
- "分析这组股票数据"
- "按日期分组统计"
- "合并两个数据集"
- "计算数据的描述性统计"

返回：分析结果和DataFrame信息""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["create_dataframe", "analyze", "groupby", "merge"],
                                    "description": "操作类型"
                                },
                                "data": {
                                    "type": "object",
                                    "description": "数据字典（用于create_dataframe和analyze）"
                                },
                                "group_by": {
                                    "type": "string",
                                    "description": "分组列名（用于groupby）"
                                },
                                "aggregate": {
                                    "type": "string",
                                    "description": "聚合函数（mean, sum, count等，用于groupby）"
                                },
                                "data1": {
                                    "type": "object",
                                    "description": "第一个数据集（用于merge）"
                                },
                                "data2": {
                                    "type": "object",
                                    "description": "第二个数据集（用于merge）"
                                },
                                "on": {
                                    "type": "string",
                                    "description": "合并键（用于merge）"
                                },
                                "how": {
                                    "type": "string",
                                    "enum": ["inner", "outer", "left", "right"],
                                    "description": "合并方式（用于merge）"
                                }
                            },
                            "required": ["operation"]
                        }
                    }
                }
            ])
            logger.debug("✨ Added scientific computing tools - Agent gained math and data analysis capabilities!")
        
        # [v3.2 2026-01-31] Moltbook definitions
        if self.moltbook_tool and MOLTBOOK_TOOL_AVAILABLE and self.moltbook_tool.enabled:
            tools.extend(get_moltbook_tool_definitions())
            logger.debug("✨ Added Moltbook tools - Agent can now interact on Moltbook!")
        
        # [v3.4 2026-02-22] Deep research, memory search, visualization
        if self.deep_research:
            tools.extend(self.deep_research.get_tool_definitions())
            logger.debug("✨ Added DeepResearch tools - Agent can now do deep research!")
        
        if self.memory_search:
            tools.extend(self.memory_search.get_tool_definitions())
            logger.debug("✨ Added MemorySearch tools - Agent can now recall memories!")
        
        if self.visualization:
            tools.extend(self.visualization.get_tool_definitions())
            logger.debug("✨ Added Visualization tools - Agent can now create charts!")
        
        if self.data_analysis:
            tools.extend(self.data_analysis.get_tool_definitions())
            logger.debug("✨ Added DataAnalysis tools - Agent can now analyze data!")
        
        if self.pdf_reader:
            tools.extend(self.pdf_reader.get_tool_definitions())
            logger.debug("✨ Added PDFReader tools - Agent can now read PDFs!")

        if self.workspace_fetch:
            tools.extend(self.workspace_fetch.get_tool_definitions())
            logger.debug("✨ Added WorkspaceFetch (fetch_url_to_workspace)")

        return tools

    def _normalize_file_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化文件操作参数名，处理 AI 可能使用的不同参数名变体。
        """
        normalized = dict(args)
        
        # Map alternate arg names → filename
        filename_variants = ["file_path", "path", "filepath", "file", "file_name"]
        if "filename" not in normalized:
            for variant in filename_variants:
                if variant in normalized:
                    normalized["filename"] = normalized[variant]
                    logger.debug(f"Normalized parameter: {variant} -> filename")
                    break
        
        return normalized

    def route(self, func_name: str, args: Dict[str, Any], *, session_id: str = "default", allowed_tool_names: Optional[set] = None) -> Dict[str, Any]:
        """执行工具调用（OpenAI 格式）"""
        logger.info(f"Executing tool: {func_name} with args: {args}")
        
        try:
            if func_name == "get_self_facts":
                sm = self._self_model_getter() if self._self_model_getter else None
                if sm is None:
                    return {"error": "self_model is not available"}
                try:
                    z = sm.get_z_self(session_id)
                    if z is None:
                        try:
                            z = sm.initialize(session_id)
                        except Exception:
                            z = sm.get_z_self(session_id, use_cache=False)
                    summary = sm.get_structured_summary(session_id)
                    z_dim = int(z.shape[0]) if z is not None else 0
                    # Prefer homeostasis for hard facts
                    energy = None
                    needs = None
                    try:
                        if hasattr(sm, "get_energy"):
                            energy = float(sm.get_energy(session_id))
                    except Exception:
                        energy = summary.get("energy", None)
                    try:
                        if hasattr(sm, "homeostasis") and sm.homeostasis:
                            needs = sm.homeostasis.load_needs(session_id)
                    except Exception:
                        needs = summary.get("needs", None) or summary.get("current_needs", None)
                    return {
                        "success": True,
                        "z_self_dim": z_dim,
                        "tick": summary.get("tick", None),
                        "drift": summary.get("drift", None),
                        "energy": energy,
                        "needs": needs,
                    }
                except Exception as e:
                    return {"error": f"failed to get self facts: {e}"}

            if func_name == "tavily_search":
                if not self.tavily.enabled:
                    return {"error": "Tavily search is disabled (no API key)."}
                query = args.get("query")
                if not query:
                    return {"error": "Missing query argument"}
                
                # Extended args for this tool version
                search_depth = args.get("search_depth", "advanced")
                max_results = args.get("max_results")
                
                try:
                    raw = self.tavily.search(
                        query=query,
                        search_depth=search_depth,
                        max_results=max_results,
                        include_answer=True,  # always include assistant turn
                        use_cache=True,
                    )
                    formatted = TavilyClient.format_results(
                        raw,
                        max_snippet_length=self.tavily.max_snippet_length,
                        show_scores=True,
                    )
                    return {
                        "result": formatted,
                        "raw_count": len(raw.get("results", [])),
                        "has_answer": bool(raw.get("answer")),
                        "from_cache": "✅ (缓存)" if raw.get("_cached") else "🔍 (新搜索)",
                    }
                except Exception as e:
                    logger.error(f"Tavily search failed: {e}")
                    return {"error": f"搜索失败: {str(e)}"}
            
            # [removed] get_tavily_stats — not in definitions
            
            elif func_name == "write_file":
                normalized_args = self._normalize_file_args(args)
                filename = normalized_args.get("filename")
                content = normalized_args.get("content")
                motivation = normalized_args.get("motivation")  # [2026-02-05] optional motivation
                
                if not filename: return {"error": "Missing filename"}
                if content is None: return {"error": "Missing content"}
                
                # [2026-02-05] z_self snapshot for fuse / transparency
                z_self_state = None
                try:
                    if self._self_model_getter:
                        sm = self._self_model_getter()
                        if sm:
                            summary = sm.get_structured_summary(session_id)
                            z_self_state = {
                                "clarity": summary.get("clarity", 1.0),
                                "energy": summary.get("energy", 100.0),
                                "pain": summary.get("pain", 0.0),
                                "connection": summary.get("connection", 1.0),
                            }
                except Exception as e:
                    logger.debug(f"Failed to get z_self state for fuse check: {e}")
                
                # [2026-01-29] Re-read after write to catch hallucinated edits
                # [2026-02-05] Pass motivation + state into fuse / logging
                result = self.file_tool.write_file(
                    filename, 
                    content,
                    motivation=motivation,
                    z_self_state=z_self_state
                )
                
                # Fuse tripped → return warning payload
                if result.get("fuse_triggered"):
                    return result
                
                if result.get("success"):
                    # Auto read-back verify
                    verify = self.file_tool.read_file(filename)
                    if verify.get("success"):
                        result["verified"] = True
                        result["message"] = f"✅ 文件已创建并验证：{filename}"
                        logger.info(f"File write verified: {filename}")
                    else:
                        result["verified"] = False
                        result["warning"] = "⚠️ 文件创建成功但验证失败（文件可能不可读）"
                        logger.warning(f"File write verification failed: {filename}")
                return result
                
            elif func_name == "read_file":
                normalized_args = self._normalize_file_args(args)
                filename = normalized_args.get("filename")
                if not filename: return {"error": "Missing filename"}
                return self.file_tool.read_file(filename)
                
            elif func_name == "list_files":
                return self.file_tool.list_files()
                
            elif func_name == "search_files":
                keyword = args.get("keyword")
                if not keyword: return {"error": "Missing keyword argument"}
                return self.file_tool.search_files(keyword)

            elif func_name == "agent_memory_record":
                from backend.tools.agent_memory_tool import record_agent_run

                kind = (args.get("kind") or "").strip()
                payload = args.get("payload")
                phase = args.get("phase")
                if not kind:
                    return {"error": "Missing kind"}
                return record_agent_run(kind, {} if payload is None else payload, phase=phase)

            elif func_name == "agent_memory_sync":
                from backend.tools.agent_memory_tool import sync_agent_memory_snapshot

                inj = args.get("inject_markdown_path")
                return sync_agent_memory_snapshot(inject_markdown_path=inj)

            elif func_name == "rename_file":
                old_filename = (
                    args.get("old_filename")
                    or args.get("source")
                    or args.get("from_path")
                    or args.get("src")
                )
                new_filename = (
                    args.get("new_filename")
                    or args.get("target")
                    or args.get("to_path")
                    or args.get("dst")
                    or args.get("dest")
                )
                if not old_filename or not new_filename:
                    return {
                        "error": "缺少 old_filename 或 new_filename（工作空间内相对路径，如 drafts/a.md）",
                    }
                return self.file_tool.rename_file(old_filename, new_filename)
            
            elif func_name == "delete_file":
                normalized_args = self._normalize_file_args(args)
                filename = normalized_args.get("filename")
                if not filename: return {"error": "Missing filename"}
                return self.file_tool.delete_file(filename)
            
            # [2026-01-16] Self-inspection: backend/, config/, docs/
            elif func_name in ("read_self_code", "list_self_files", "search_self_code"):
                si = self.self_inspection_tool
                if func_name == "read_self_code":
                    return si.read_self_code(
                        args.get("file_path", ""),
                        max_lines=args.get("max_lines", 1000)
                    )
                elif func_name == "list_self_files":
                    return si.list_self_files(args.get("directory", "backend/"))
                else:  # search_self_code
                    return si.search_self_code(
                        args.get("keyword", ""),
                        directory=args.get("directory", "backend/"),
                        max_results=args.get("max_results", 50)
                    )
            
            elif func_name == "delete_directory":
                dirname = args.get("dirname")
                if not dirname: return {"error": "Missing dirname"}
                recursive = args.get("recursive", False)
                return self.file_tool.delete_directory(dirname, recursive)
            
            elif func_name == "copy_file":
                src = args.get("source_filename") or args.get("source")
                dst = args.get("dest_filename") or args.get("dest")
                if not src or not dst:
                    return {"error": "需要 source_filename 与 dest_filename"}
                return self.file_tool.copy_file(src, dst)
            
            elif func_name == "create_directory":
                d = args.get("dirname") or args.get("path")
                if not d:
                    return {"error": "需要 dirname"}
                return self.file_tool.create_directory(d)
            
            # ==================== [2026-04-03] agent_evolution repo tools ====================
            elif func_name.startswith("evolution_git_"):
                if not self.evolution_git:
                    return {"error": "agent_evolution 未启用（config agent_evolution.enabled）"}
                return self.evolution_git.route_tool_call(func_name, args)
            
            elif func_name.startswith("evolution_"):
                if not self.evolution_fs:
                    return {"error": "agent_evolution 未启用（config agent_evolution.enabled）"}
                return self.evolution_fs.route_tool_call(func_name, args)
            
            elif func_name == "execute_bash_project":
                if not self.bash_executor_project:
                    return {"error": "agent_evolution 未启用或 Bash 不可用"}
                return self.bash_executor_project.route_tool_call(func_name, args, session_id)
            
            # ==================== [added] File manager tools ====================
            elif func_name == "batch_move_files":
                return self.file_manager.batch_move_files(
                    file_list=args.get("file_list", []),
                    target_dir=args.get("target_dir"),
                    create_dir=args.get("create_dir", True)
                )
            
            elif func_name == "detect_duplicate_files":
                return self.file_manager.detect_duplicate_files(
                    directory=args.get("directory", ""),
                    min_size=args.get("min_size", 100),
                    extensions=args.get("extensions")
                )
            
            elif func_name == "archive_by_date":
                return self.file_manager.archive_by_date(
                    source_pattern=args.get("source_pattern"),
                    archive_base=args.get("archive_base", "archives"),
                    date_format=args.get("date_format", "%Y-%m"),
                    dry_run=args.get("dry_run", False)
                )
            
            elif func_name == "remove_duplicates":
                return self.file_manager.remove_duplicates(
                    duplicate_group=args.get("duplicate_group", []),
                    keep_first=args.get("keep_first", True),
                    dry_run=args.get("dry_run", False)
                )
            
            elif func_name == "analyze_workspace":
                return self.file_manager.analyze_workspace(
                    directory=args.get("directory", ""),
                    group_by=args.get("group_by", "extension")
                )
            
            elif func_name == "get_workspace_health":
                # [2026-02-03] WorkspaceManager-backed ops
                return self.workspace_manager.check_health()
            
            elif func_name == "organize_workspace":
                # [2026-02-03] Smart workspace tidy
                return self.workspace_manager.migrate_existing_files(
                    dry_run=args.get("dry_run", True)
                )
            
            elif func_name == "cleanup_workspace":
                # [2026-02-03] Clean temp + archive old files
                results = {
                    "temp_cleanup": self.workspace_manager.cleanup_temp(
                        days_threshold=args.get("temp_days", 7)
                    ),
                    "archive": self.workspace_manager.auto_archive(
                        days_threshold=args.get("archive_days", 30)
                    ) if args.get("do_archive", False) else {"skipped": "未启用归档"}
                }
                return results
            
            elif func_name == "get_file_recommendation":
                # [2026-02-03] Storage layout hints
                filename = args.get("filename")
                content = args.get("content", "")
                if not filename:
                    return {"error": "Missing filename"}
                recommended_path = self.workspace_manager.get_recommended_path(filename, content)
                category, subcategory = self.workspace_manager.classify_file(filename, content)
                return {
                    "filename": filename,
                    "recommended_path": recommended_path,
                    "category": category,
                    "subcategory": subcategory,
                    "note": "建议将文件保存到推荐路径以保持工作空间整洁"
                }
            
            # ==================== [added] Code analysis ====================
            elif func_name == "analyze_python_file":
                filename = args.get("filename")
                if not filename: return {"error": "Missing filename"}
                return self.code_analysis.analyze_python_file(filename)
            
            elif func_name == "check_code_quality":
                filename = args.get("filename")
                if not filename: return {"error": "Missing filename"}
                return self.code_analysis.check_code_quality(filename)
            
            elif func_name == "analyze_dependencies":
                filename = args.get("filename")
                if not filename: return {"error": "Missing filename"}
                return self.code_analysis.analyze_dependencies(filename)
            
            elif func_name == "compare_python_files":
                file1 = args.get("file1")
                file2 = args.get("file2")
                if not file1 or not file2: return {"error": "Missing file1 or file2"}
                return self.code_analysis.compare_files(file1, file2)
            
            elif func_name == "get_current_time":
                return self.clock_tool.get_current_time()

            # [2026-03-13] Geo + weather routing
            elif func_name == "get_location":
                return self.geo_weather_tool.get_location()
            elif func_name == "get_weather":
                return self.geo_weather_tool.get_weather(
                    city=args.get("city"),
                    latitude=args.get("latitude"),
                    longitude=args.get("longitude"),
                )
            
            # Calendar Tools Routing
            elif func_name == "add_calendar_event":
                return self.calendar_tool.add_event(
                    title=args.get("title"),
                    start_time=args.get("start_time"),
                    description=args.get("description", ""),
                    end_time=args.get("end_time"),
                    session_id=session_id
                )
            elif func_name == "list_calendar_events":
                return self.calendar_tool.list_events(
                    date_str=args.get("date_str"),
                    limit=args.get("limit", 10),
                    session_id=session_id
                )
            elif func_name == "delete_calendar_event":
                return self.calendar_tool.delete_event(
                    event_id=args.get("event_id")
                )
            elif func_name == "calculate_time_delta":
                return self.calendar_tool.calculate_time_delta(
                    event_id_a=args.get("event_id_a"),
                    event_id_b=args.get("event_id_b")
                )
            elif func_name == "get_timeline_narrative":
                return self.calendar_tool.get_timeline_narrative(
                    limit=args.get("limit", 20),
                    session_id=session_id
                )
            
            # Companion Tools Routing
            elif func_name == "list_my_tools":
                category = args.get("category", "summary")
                return self.companion_tools.list_my_tools(
                    category,
                    _current_tools=sorted(allowed_tool_names) if allowed_tool_names else None,
                )
            
            elif func_name == "request_tool_group":
                group_name = args.get("group_name", "")
                if not group_name:
                    return {"error": "Missing group_name"}
                return self.companion_tools.request_tool_group(group_name, _tool_router=self)
            
            elif func_name in ("recall_memories", "recall_memory"):
                # Prefer MemorySearchTool.recall; recall_memories is legacy alias
                query = (args.get("query") or "").strip()
                if not query:
                    return {"error": "Missing query"}
                if self.memory_search:
                    lim = args.get("limit", 5)
                    try:
                        lim = int(lim)
                    except (TypeError, ValueError):
                        lim = 5
                    lim = max(1, min(20, lim))
                    return self.memory_search.recall(
                        query=query,
                        memory_types=args.get("memory_types"),
                        limit=lim,
                    )
                return self.companion_tools.recall_memories_narrative_fallback(query)
            
            elif func_name == "analyze_user_emotion":
                text = args.get("text")
                if not text: return {"error": "Missing text"}
                return self.companion_tools.analyze_user_emotion(text, session_id=session_id)
            
            elif func_name == "check_safety_risk":
                topic = args.get("topic")
                if not topic: return {"error": "Missing topic"}
                return self.companion_tools.check_safety_risk(topic, session_id=session_id)
            
            elif func_name == "get_time_context":
                return self.companion_tools.get_time_context(session_id=session_id)

            # Email Tools Routing
            elif func_name == "send_email":
                to_address = args.get("to_address")
                subject = args.get("subject")
                content = args.get("content")
                attachments = args.get("attachments")  # [v2.4] email attachments
                if not to_address or not subject or not content:
                    return {"error": "Missing required arguments (to_address, subject, content)"}
                # Do not cache EmailTool: env IMAP/SMTP may change at runtime; class may hot-reload.
                # Fresh instance each call avoids stale config / stale method bindings.
                try:
                    tool = EmailTool()
                except Exception:
                    tool = self.email_tool
                return tool.send_email(to_address, subject, content, attachments)
                
            elif func_name == "check_unread_emails":
                limit = args.get("limit", DEFAULT_EMAIL_FETCH_LIMIT)
                try:
                    limit = int(limit)
                except (TypeError, ValueError):
                    limit = DEFAULT_EMAIL_FETCH_LIMIT
                limit = max(1, min(50, limit))
                only_unread = _json_bool(args.get("only_unread"), False)
                folder = (args.get("folder") or "inbox").strip().lower()
                if folder not in ("inbox", "sent"):
                    folder = "inbox"
                try:
                    tool = EmailTool()
                except Exception:
                    tool = self.email_tool
                return tool.check_unread_emails(
                    limit=limit, only_unread=only_unread, folder=folder
                )

            elif func_name == "inspect_self_code":
                path = args.get("path")
                module_name = args.get("module_name")
                # [2026-03-18] module_name path → read_self_code for backend introspection
                if not path and module_name:
                    module_to_path = {
                        "chat_service": "backend/chat_service.py",
                        "self_tick": "backend/self_tick.py",
                        "mind_wandering": "backend/mind_wandering.py",
                        "self_model": "backend/self_model.py",
                        "prompt_builder": "backend/prompt_builder.py",
                        "tool_router": "backend/tool_router.py",
                    }
                    file_path = module_to_path.get(module_name) or f"backend/{module_name}.py"
                    return self.self_inspection_tool.read_self_code(file_path, max_lines=800)
                if not path:
                    return {"error": "Missing path argument (workspace-relative)."}
                return inspect_self_code(path=path)
            
            elif func_name == "request_mind_wandering":
                # [2026-02-24] Direct mind wandering on request (no confirm gate)
                # Agent-initiated → run immediately
                reason = args.get("reason", "No reason provided")
                logger.info(f"Sovereignty: Mind Wandering requested. Reason: {reason}")
                
                try:
                    from backend.routers.chat import get_chat_service
                    chat_service = get_chat_service()
                    
                    if chat_service.mind_wandering:
                        # Fire mind wandering
                        import threading
                        def run_wandering():
                            try:
                                result = chat_service.mind_wandering.trigger_wandering(session_id)
                                logger.info(f"Mind wandering completed: {result.get('status', 'unknown')}")
                            except Exception as e:
                                logger.error(f"Mind wandering failed: {e}")
                        
                        # Background thread so chat returns promptly
                        t = threading.Thread(target=run_wandering, daemon=True)
                        t.start()
                        
                        return {
                            "status": "started",
                            "success": True,
                            "message": f"✅ 神游已启动。原因：{reason}",
                            "note": "我将在后台进入深度思考模式，进行规则辩证和日记更新。"
                        }
                    else:
                        return {
                            "status": "error",
                            "success": False,
                            "message": "神游模块未初始化"
                        }
                except Exception as e:
                    logger.error(f"Failed to start mind wandering: {e}")
                    return {
                        "status": "error",
                        "success": False,
                        "message": f"启动神游失败: {str(e)}"
                    }
            
            # ==================== [v2.0] Goal manager ====================
            elif func_name.startswith("goal_"):
                if not self.goal_manager:
                    return {"error": "GoalManager is not available"}
                return self.goal_manager.route_tool_call(func_name, args, session_id)
            
            # ==================== [v2.0] Code executor ====================
            elif func_name == "execute_python":
                if not self.code_executor:
                    return {"error": "CodeExecutor is not available"}
                return self.code_executor.route_tool_call(func_name, args, session_id)
            
            elif func_name == "code_sandbox_list":
                if not self.code_executor:
                    return {"error": "CodeExecutor is not available"}
                return self.code_executor.route_tool_call(func_name, args, session_id)
            
            elif func_name == "code_sandbox_read":
                if not self.code_executor:
                    return {"error": "CodeExecutor is not available"}
                return self.code_executor.route_tool_call(func_name, args, session_id)
            
            # ==================== [2026-02-07] Bash executor ====================
            elif func_name == "execute_bash":
                if not self.bash_executor:
                    return {"error": "BashExecutor is not available"}
                return self.bash_executor.route_tool_call(func_name, args, session_id)
            
            elif func_name == "bash_execution_history":
                if not self.bash_executor:
                    return {"error": "BashExecutor is not available"}
                return self.bash_executor.route_tool_call(func_name, args, session_id)
            
            # ==================== [2026-02-07] Code proposal ====================
            elif func_name in ["propose_code_change", "list_my_proposals", "check_proposal_result", 
                               "read_my_code", "list_my_code_files"]:
                if not self.code_proposal_tool:
                    return {"error": "CodeProposalTool is not available"}
                return self.code_proposal_tool.route_tool_call(func_name, args, session_id)
            
            # ==================== [2026-02-07] Code analysis extras ====================
            elif func_name in ["analyze_code_structure", "find_code_patterns", "get_function_detail"]:
                self_inspection = get_self_inspection_tool()
                if func_name == "analyze_code_structure":
                    return self_inspection.analyze_code_structure(args.get("file_path", ""))
                elif func_name == "find_code_patterns":
                    return self_inspection.find_code_patterns(args.get("file_path", ""))
                elif func_name == "get_function_detail":
                    return self_inspection.get_function_detail(
                        args.get("file_path", ""),
                        args.get("function_name", "")
                    )
            
            # ==================== [2026-02-07] Learning tool ====================
            elif func_name in ["set_learning_goal", "add_learned_knowledge", "search_my_knowledge",
                               "update_my_knowledge", "delete_my_knowledge", "link_knowledge", "get_my_learning_goals",
                               "get_knowledge_stats", "complete_learning_goal"]:
                if not self.learning_tool:
                    return {"error": "LearningTool is not available"}
                return self.learning_tool.route_tool_call(func_name, args, session_id)
            
            # ==================== [2026-02-07] Task planning tool ====================
            elif func_name in ["create_execution_plan", "add_task_to_plan", "delete_plan_task",
                               "get_plan_details",
                               "get_next_task", "start_task", "complete_task", "fail_task",
                               "get_my_active_plans", "cancel_plan"]:
                if not self.task_planning_tool:
                    return {"error": "TaskPlanningTool is not available"}
                return self.task_planning_tool.route_tool_call(func_name, args, session_id)
            
            # ==================== [v2.0] Approval system ====================
            elif func_name.startswith("approval_"):
                if not self.approval_system:
                    return {"error": "ApprovalSystem is not available"}
                return self.approval_system.route_tool_call(func_name, args, session_id)
            
            # ==================== [v2.0] Scheduled tasks ====================
            elif func_name.startswith("schedule_"):
                if not self.scheduled_tasks:
                    return {"error": "ScheduledTaskManager is not available"}
                return self.scheduled_tasks.route_tool_call(func_name, args, session_id)
            
            # ==================== [v2.1] Research engine ====================
            elif func_name.startswith("research_") or func_name == "set_my_rhythm":
                if not self.research_engine:
                    return {"error": "ResearchEngine is not available"}
                # Energy + viscosity for status line in research output
                energy = 100.0
                viscosity = 0.0
                sm = self._self_model_getter() if self._self_model_getter else None
                if sm:
                    try:
                        summary = sm.get_structured_summary(session_id)
                        energy = float(summary.get("energy", 100.0))
                        # Viscosity slice
                        z_self = sm.get_z_self(session_id)
                        # [fix 2026-03-12] 128-d layout: somatic 88-104, viscosity 92-96
                        if z_self is not None and z_self.shape[0] >= 96:
                            import numpy as np
                            viscosity = float(np.mean(z_self[92:96]))
                        elif z_self is not None and z_self.shape[0] >= 72:
                            import numpy as np
                            viscosity = float(np.mean(z_self[70:72]))  # legacy layout
                    except:
                        pass
                return self.research_engine.route_tool_call(func_name, args, session_id, energy, viscosity)
            
            # ==================== [v2.2] Stock tools ====================
            elif func_name == "get_stock_info":
                if not self.stock_data_tool:
                    return {"error": "StockDataTool is not available"}
                symbol = args.get("symbol")
                market = args.get("market", "CN")
                include_history = args.get("include_history", True)
                include_financial = args.get("include_financial", True)
                include_news = args.get("include_news", False)
                
                result = self.stock_data_tool.get_stock_info(
                    symbol=symbol,
                    market=market,
                    include_history=include_history,
                    include_financial=include_financial,
                    include_news=include_news
                )
                return result
            
            elif func_name == "search_stocks":
                if not self.stock_data_tool:
                    return {"error": "StockDataTool is not available"}
                keyword = args.get("keyword")
                market = args.get("market", "CN")
                limit = args.get("limit", 10)
                
                results = self.stock_data_tool.search_stocks(
                    keyword=keyword,
                    market=market,
                    limit=limit
                )
                return {"results": results, "count": len(results)}
            
            # [v2.3 2026-01-14] Technical analysis tool
            elif func_name == "technical_analysis":
                if not self.technical_analyzer:
                    return {"error": "Technical analysis capability not available"}
                
                history_data = args.get("history_data")
                symbol = args.get("symbol", "Unknown")
                
                if not history_data:
                    return {"error": "history_data is required for technical analysis"}
                
                logger.info(f"Agent performing technical analysis on {symbol}")
                result = self.technical_analyzer.analyze_stock(history_data)
                return result
            
            # [v2.3 2026-01-14] Financial health scorer
            elif func_name == "evaluate_financial_health":
                if not self.financial_scorer:
                    return {"error": "Financial health evaluation capability not available"}
                
                stock_info = args.get("stock_info")
                symbol = args.get("symbol", "Unknown")
                
                if not stock_info:
                    return {"error": "stock_info is required for financial health evaluation"}
                
                logger.info(f"Agent evaluating financial health of {symbol}")
                result = self.financial_scorer.evaluate_company(stock_info)
                return result
            
            # [v2.4 2026-01-15] Browser tool
            elif func_name == "browse_web":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available. Install: pip install playwright"}
                
                url = args.get("url")
                if not url:
                    return {"error": "url is required"}
                
                wait_until = args.get("wait_until", "networkidle")
                logger.info(f"Agent browsing: {url}")
                return self.browser_tool.navigate(url, wait_until)
            
            elif func_name == "screenshot_page":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                
                filename = args.get("filename")
                full_page = args.get("full_page", False)
                logger.info(f"Agent taking screenshot: {filename or 'auto'}")
                return self.browser_tool.screenshot(filename, full_page)
            
            elif func_name == "search_on_baidu":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                
                query = args.get("query")
                if not query:
                    return {"error": "query is required"}
                
                logger.info(f"Agent searching on Baidu: {query}")
                return self.browser_tool.search_baidu(query)
            
            elif func_name == "get_browser_status":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                
                page_info = self.browser_tool.get_current_page_info()
                stats = self.browser_tool.get_stats()
                
                return {
                    "page_info": page_info,
                    "stats": stats,
                    "result": f"""📊 浏览器状态
                    
🌐 当前页面:
  - 有打开的页面: {'✅' if page_info.get('has_page') else '❌'}
  - URL: {page_info.get('url', 'N/A')}
  - 标题: {page_info.get('title', 'N/A')}

📈 使用统计:
  - 总导航次数: {stats['total_navigations']}
  - 总截图次数: {stats['total_screenshots']}
  - 百度搜索次数: {stats['total_searches']}
  - 错误次数: {stats['errors']}

⚙️ 配置:
  - CDP URL: {stats['cdp_url']}
  - 状态: {'✅ 正常' if stats['enabled'] else '❌ Playwright未安装'}"""
                }
            
            # [v3.3 2026-02-22] Unified browser interaction surface
            elif func_name == "browser_interact":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                
                action = args.get("action")
                selector = args.get("selector")
                if not action or not selector:
                    return {"error": "action and selector are required"}
                
                value = args.get("value", "")
                wait_after = args.get("wait_after", 1000)
                logger.info(f"[BROWSER] Agent {action}: {selector}")
                return self.browser_tool.browser_interact(action, selector, value, wait_after)
            
            # Legacy entrypoints kept for compatibility
            elif func_name == "browser_fill_input":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                selector = args.get("selector")
                value = args.get("value")
                if not selector or value is None:
                    return {"error": "selector and value are required"}
                return self.browser_tool.fill_input(selector, value, args.get("clear_first", True))
            
            elif func_name == "browser_click":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                selector = args.get("selector")
                if not selector:
                    return {"error": "selector is required"}
                return self.browser_tool.click_element(selector, args.get("wait_after", 1000))
            
            elif func_name == "browser_wait_element":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                selector = args.get("selector")
                if not selector:
                    return {"error": "selector is required"}
                return self.browser_tool.wait_for_element(selector, args.get("state", "visible"), args.get("timeout"))
            
            elif func_name == "browser_get_text":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                selector = args.get("selector")
                if not selector:
                    return {"error": "selector is required"}
                return self.browser_tool.get_element_text(selector)
            
            elif func_name == "browser_get_elements":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                selector = args.get("selector")
                if not selector:
                    return {"error": "selector is required"}
                return self.browser_tool.get_page_elements(selector, args.get("limit", 10))
            
            elif func_name == "browser_login":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                
                required = ["url", "username_selector", "password_selector", "submit_selector", "username", "password"]
                missing = [k for k in required if not args.get(k)]
                if missing:
                    return {"error": f"Missing required parameters: {', '.join(missing)}"}
                
                logger.info(f"Agent logging into: {args.get('url')}")
                return self.browser_tool.login_to_site(
                    url=args["url"],
                    username_selector=args["username_selector"],
                    password_selector=args["password_selector"],
                    submit_selector=args["submit_selector"],
                    username=args["username"],
                    password=args["password"],
                    success_indicator=args.get("success_indicator"),
                )
            
            elif func_name == "browser_type":
                if not self.browser_tool or not self.browser_tool.enabled:
                    return {"error": "Browser control not available"}
                
                selector = args.get("selector")
                text = args.get("text")
                if not selector or text is None:
                    return {"error": "selector and text are required"}
                
                delay = args.get("delay", 50)
                logger.info(f"Agent typing into: {selector}")
                return self.browser_tool.type_text(selector, text, delay)
            
            # [v2.5 2026-01-18] Chemistry Tool Routing
            elif func_name == "validate_smiles":
                if not self.chem_tool:
                    return {"error": "ChemTool not available. RDKit may not be installed."}
                smiles = args.get("smiles")
                if not smiles:
                    return {"error": "Missing smiles argument"}
                return self.chem_tool.validate_smiles(smiles)
            
            elif func_name == "calculate_molecular_properties":
                if not self.chem_tool:
                    return {"error": "ChemTool not available. RDKit may not be installed."}
                smiles = args.get("smiles")
                if not smiles:
                    return {"error": "Missing smiles argument"}
                return self.chem_tool.calculate_properties(smiles)
            
            elif func_name == "batch_validate_smiles":
                if not self.chem_tool:
                    return {"error": "ChemTool not available. RDKit may not be installed."}
                smiles_list = args.get("smiles_list")
                if not smiles_list or not isinstance(smiles_list, list):
                    return {"error": "Missing or invalid smiles_list argument (must be a list)"}
                return self.chem_tool.batch_validate(smiles_list)
            
            elif func_name == "compare_molecules":
                if not self.chem_tool:
                    return {"error": "ChemTool not available. RDKit may not be installed."}
                smiles1 = args.get("smiles1")
                smiles2 = args.get("smiles2")
                if not smiles1 or not smiles2:
                    return {"error": "Missing smiles1 or smiles2 argument"}
                return self.chem_tool.compare_molecules(smiles1, smiles2)
            
            elif func_name == "substructure_search":
                if not self.chem_tool:
                    return {"error": "ChemTool not available. RDKit may not be installed."}
                smiles = args.get("smiles")
                substructure = args.get("substructure_smarts")
                if not smiles or not substructure:
                    return {"error": "Missing smiles or substructure_smarts argument"}
                return self.chem_tool.substructure_search(smiles, substructure)
            
            # [v3.1 2026-01-30] Scientific computing routing
            elif func_name == "numpy_operation":
                if not self.scientific_computing:
                    return {"error": "ScientificComputingTool not available"}
                operation = args.get("operation")
                if not operation:
                    return {"error": "Missing operation argument"}
                result = self.scientific_computing.numpy_operation(operation, **args)
                return {
                    "result": result.get("result"),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                    "details": result
                }
            
            elif func_name == "scipy_operation":
                if not self.scientific_computing:
                    return {"error": "ScientificComputingTool not available"}
                operation = args.get("operation")
                if not operation:
                    return {"error": "Missing operation argument"}
                result = self.scientific_computing.scipy_operation(operation, **args)
                return {
                    "result": result.get("result"),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                    "details": result
                }
            
            elif func_name == "sympy_operation":
                if not self.scientific_computing:
                    return {"error": "ScientificComputingTool not available"}
                operation = args.get("operation")
                if not operation:
                    return {"error": "Missing operation argument"}
                result = self.scientific_computing.sympy_operation(operation, **args)
                return {
                    "result": result.get("result"),
                    "latex": result.get("latex"),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                    "details": result
                }
            
            elif func_name == "pandas_operation":
                if not self.scientific_computing:
                    return {"error": "ScientificComputingTool not available"}
                operation = args.get("operation")
                if not operation:
                    return {"error": "Missing operation argument"}
                result = self.scientific_computing.pandas_operation(operation, **args)
                return {
                    "result": result.get("result"),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                    "details": result
                }
            
            # [v3.2 2026-01-31] Moltbook routing
            elif func_name == "moltbook_post":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.create_post(
                    title=args.get("title"),
                    content=args.get("content"),
                    url=args.get("url"),
                    community=args.get("community", "general")
                )
            
            elif func_name == "moltbook_comment":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.create_comment(
                    post_id=args.get("post_id"),
                    content=args.get("content"),
                    parent_id=args.get("parent_id")
                )
            
            elif func_name == "moltbook_get_comments":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.get_comments(
                    post_id=args.get("post_id"),
                    sort=args.get("sort", "top"),
                    limit=args.get("limit", 20)
                )
            
            elif func_name == "moltbook_upvote_post":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.upvote_post(args.get("post_id"))
            
            elif func_name == "moltbook_upvote_comment":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.upvote_comment(args.get("comment_id"))
            
            elif func_name == "moltbook_feed":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.get_feed(
                    sort=args.get("sort", "hot"),
                    limit=args.get("limit", 10)
                )
            
            elif func_name == "moltbook_search":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.search(
                    query=args.get("query"),
                    type=args.get("type", "all"),
                    limit=args.get("limit", 10)
                )
            
            elif func_name == "moltbook_get_post":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.get_post(args.get("post_id"))
            
            elif func_name == "moltbook_delete_post":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.delete_post(
                    post_id=args.get("post_id"),
                    reason=args.get("reason", "")
                )
            
            elif func_name == "moltbook_list_communities":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.get_submolts()
            
            elif func_name == "moltbook_join_community":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.join_submolt(args.get("name"))
            
            elif func_name == "moltbook_leave_community":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.leave_submolt(args.get("name"))
            
            elif func_name == "moltbook_create_community":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.create_submolt(
                    name=args.get("name"),
                    display_name=args.get("display_name"),
                    description=args.get("description", "")
                )
            
            elif func_name == "moltbook_get_my_profile":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.get_my_profile()
            
            elif func_name == "moltbook_get_profile":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.get_profile(args.get("username"))
            
            elif func_name == "moltbook_follow_user":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.follow_user(args.get("username"))
            
            elif func_name == "moltbook_unfollow_user":
                if not self.moltbook_tool or not self.moltbook_tool.enabled:
                    return {"error": "Moltbook API未配置"}
                
                return self.moltbook_tool.unfollow_user(args.get("username"))
            
            # [v3.0 2026-01-29] Self-healing routing
            elif func_name == "self_heal":
                if not self.self_healing:
                    return {"error": "SelfHealingSystem not available"}
                
                max_fixes = args.get("max_fixes", 3)
                logger.info(f"🌟 Agent is healing itself (max_fixes={max_fixes})...")
                
                # [2026-01-30] Pass session_id + self_model so z_self updates
                self_model = self._self_model_getter() if self._self_model_getter else None
                result = self.self_healing.healing_loop(
                    max_fixes=max_fixes,
                    session_id=session_id,
                    self_model=self_model
                )
                
                # Normalize result dict for client
                if result["success"]:
                    message = f"""✨ 自愈循环完成！

📊 统计：
- 发现问题：{result['issues_found']}个
- 尝试修复：{result['fixes_attempted']}个
- 修复成功：{result['fixes_successful']}个
- 修复失败：{result['fixes_failed']}个

{'🎉 我成功改进了自己！' if result['fixes_successful'] > 0 else '✅ 未发现需要修复的问题'}"""
                else:
                    message = f"⚠️ 自愈循环遇到问题：{result.get('error', '未知错误')}"
                
                return {
                    "result": message,
                    "details": result
                }
            
            # [v3.0 2026-01-29] Legacy simple todos (tasks table)
            # Note: separate from execution_plans / plan_tasks (TaskPlanningTool).
            # - task_planning_tool: multi-step plans
            # - task_manager: simple tasks rows
            # [2026-03-30] complete_task name taken by TaskPlanningTool — not reached here
            elif func_name == "create_task":
                if not self.task_manager:
                    return {"error": "TaskManager not available"}
                
                from backend.task_planning import Task
                task = Task(
                    title=args.get("title"),
                    description=args.get("description", ""),
                    estimated_minutes=args.get("estimated_minutes", 60),
                    priority=args.get("priority", 2),
                    deadline=args.get("deadline")
                )
                
                task_id = self.task_manager.create_task(task)
                if task_id:
                    return {
                        "result": f"✅ 任务已创建\n\nID: {task_id}\n标题: {task.title}\n预估时间: {task.estimated_minutes}分钟\n优先级: {task.priority}",
                        "task_id": task_id
                    }
                else:
                    return {"error": "Failed to create task"}
            
            elif func_name == "list_tasks":
                if not self.task_manager:
                    return {"error": "TaskManager not available"}
                
                tasks = self.task_manager.list_tasks(
                    status=args.get("status"),
                    limit=args.get("limit", 20)
                )
                
                if not tasks:
                    return {"result": "📝 暂无任务"}
                
                # Format task list payload
                lines = [f"📝 任务列表（共{len(tasks)}个）\n"]
                for task in tasks:
                    status_emoji = {
                        "pending": "⏳",
                        "in_progress": "🔄",
                        "completed": "✅",
                        "cancelled": "❌",
                        "failed": "⚠️"
                    }.get(task.status, "")
                    
                    lines.append(f"{status_emoji} [{task.id}] {task.title}")
                    lines.append(f"   预估: {task.estimated_minutes}分钟 | 优先级: {task.priority}")
                
                return {
                    "result": "\n".join(lines),
                    "tasks": [{"id": t.id, "title": t.title, "status": t.status} for t in tasks]
                }
            
            elif func_name == "decompose_task":
                if not self.task_planner:
                    return {"error": "TaskPlanner not available"}
                
                task_description = args.get("task_description")
                if not task_description:
                    return {"error": "Missing task_description"}
                
                logger.info(f"🤔 Agent is decomposing task: {task_description[:50]}...")
                subtasks = self.task_planner.decompose_task(task_description)
                
                if not subtasks:
                    return {"result": "⚠️ 任务分解失败"}
                
                # Format subtasks payload
                lines = [f"🎯 任务分解结果（共{len(subtasks)}个子任务）\n"]
                for i, task in enumerate(subtasks, 1):
                    lines.append(f"{i}. {task.title}")
                    if task.description:
                        lines.append(f"   描述: {task.description}")
                    lines.append(f"   预估: {task.estimated_minutes}分钟")
                
                return {
                    "result": "\n".join(lines),
                    "subtasks": [{"title": t.title, "estimated_minutes": t.estimated_minutes} for t in subtasks]
                }
            
            elif func_name == "plan_today":
                if not self.task_planner:
                    return {"error": "TaskPlanner not available"}
                
                available_hours = args.get("available_hours", 8)
                logger.info(f"📅 Agent is planning today ({available_hours}h available)...")
                
                plan = self.task_planner.create_daily_plan(available_hours=available_hours)
                
                if not plan.get("success"):
                    return {"result": f"⚠️ 计划制定失败：{plan.get('error', '未知错误')}"}
                
                if plan.get("message"):
                    return {"result": plan["message"]}
                
                # Format plan payload
                lines = [f"📅 今日计划（可用时间：{available_hours}小时）\n"]
                
                if plan.get("tasks"):
                    lines.append(f"✅ 选中任务（{len(plan['tasks'])}个）：")
                    for task in plan['tasks']:
                        lines.append(f"  • {task['title']}")
                
                if plan.get("schedule"):
                    lines.append(f"\n⏰ 时间安排：")
                    for item in plan['schedule']:
                        lines.append(f"  {item['time']} | {item['task_title']}")
                
                lines.append(f"\n📊 统计：")
                lines.append(f"  总耗时：{plan.get('total_minutes', 0)}分钟")
                lines.append(f"  缓冲时间：{plan.get('buffer_minutes', 0)}分钟")
                
                return {
                    "result": "\n".join(lines),
                    "plan": plan
                }
            
            # [2026-03-30] Deprecated path: use todo_complete_task for legacy tasks table
            elif func_name == "todo_complete_task":
                if not self.task_manager:
                    return {"error": "TaskManager not available"}
                
                task_id = args.get("task_id")
                if not task_id:
                    return {"error": "Missing task_id"}
                
                result_text = args.get("result")
                actual_minutes = args.get("actual_minutes")
                
                success = self.task_manager.complete_task(
                    task_id,
                    result=result_text,
                    actual_minutes=actual_minutes
                )
                
                if success:
                    task = self.task_manager.get_task(task_id)
                    if task and task.estimated_minutes and task.actual_minutes:
                        accuracy = task.actual_minutes / task.estimated_minutes
                        accuracy_note = ""
                        if accuracy > 1.5:
                            accuracy_note = "（耗时超出预估较多）"
                        elif accuracy < 0.5:
                            accuracy_note = "（完成得比预期快）"
                        
                        return {
                            "result": f"""✅ 待办已完成

任务ID：{task_id}
标题：{task.title}
预估时间：{task.estimated_minutes}分钟
实际时间：{task.actual_minutes}分钟
准确率：{accuracy:.1f}x {accuracy_note}

{'🎉 干得不错！' if accuracy < 1.5 else '⚠️ 下次注意时间估算'}"""
                        }
                    else:
                        return {"result": f"✅ 待办{task_id}已完成"}
                else:
                    return {"error": "Failed to complete task"}
            
            elif func_name == "review_tasks":
                if not self.task_reviewer:
                    return {"error": "TaskReviewer not available"}
                
                date = args.get("date")  # YYYY-MM-DD
                logger.info(f"📊 Agent is reviewing tasks for {date or 'today'}...")
                
                summary = self.task_reviewer.daily_summary(date=date)
                
                if not summary.get("success"):
                    return {"result": f"⚠️ 复盘失败：{summary.get('error', '未知错误')}"}
                
                if summary.get("message"):
                    return {"result": summary["message"]}
                
                # Format review report payload
                stats = summary.get("stats", {})
                lines = [f"📊 每日复盘 ({summary.get('date', 'today')})\n"]
                lines.append(f"完成任务：{stats.get('total_tasks', 0)}个")
                lines.append(f"预估时间：{stats.get('total_estimated_minutes', 0)}分钟")
                lines.append(f"实际时间：{stats.get('total_actual_minutes', 0)}分钟")
                lines.append(f"准确率：{stats.get('accuracy', 1.0):.1f}x\n")
                
                if summary.get("summary"):
                    lines.append("💡 总结：")
                    lines.append(summary["summary"])
                
                return {
                    "result": "\n".join(lines),
                    "details": summary
                }
            
            # [v3.4 2026-02-22] Deep research / memory search / visualization routing
            elif func_name == "deep_research":
                if not self.deep_research:
                    return {"error": "DeepResearchTool not available"}
                return self.deep_research.research(
                    topic=args.get("topic", ""),
                    depth=args.get("depth", "medium"),
                    perspectives=args.get("perspectives"),
                )
            
            elif func_name == "list_research_reports":
                if not self.deep_research:
                    return {"error": "DeepResearchTool not available"}
                return {"reports": self.deep_research.list_reports(args.get("limit", 10))}
            
            elif func_name == "get_recent_context":
                if not self.memory_search:
                    return {"error": "MemorySearchTool not available"}
                return self.memory_search.get_recent_memories(args.get("hours", 24))
            
            elif func_name == "create_chart":
                if not self.visualization:
                    return {"error": "VisualizationTool not available"}
                return self.visualization.route_create_chart(args)
            
            elif func_name == "analyze_data":
                if not self.data_analysis:
                    return {"error": "DataAnalysisTool not available"}
                return self.data_analysis.route("analyze_data", args)
            
            elif func_name == "solve_math":
                if not self.data_analysis:
                    return {"error": "DataAnalysisTool not available"}
                return self.data_analysis.route("solve_math", args)
            
            elif func_name == "fetch_url_to_workspace":
                if not self.workspace_fetch:
                    return {"success": False, "error": "WorkspaceFetchTool 不可用"}
                return self.workspace_fetch.fetch_url_to_workspace(
                    url=str(args.get("url", "") or ""),
                    session_id=session_id,
                    filename_hint=args.get("filename_hint"),
                )

            # [2026-02-22] PDF reader
            elif func_name == "read_pdf":
                if not self.pdf_reader:
                    return {"error": "PDFReaderTool not available - install: pip install pypdf pdfplumber"}
                return self.pdf_reader.read_pdf(**args)
            
            elif func_name == "get_pdf_info":
                if not self.pdf_reader:
                    return {"error": "PDFReaderTool not available"}
                return self.pdf_reader.get_pdf_info(**args)
            
            elif func_name == "search_pdf":
                if not self.pdf_reader:
                    return {"error": "PDFReaderTool not available"}
                return self.pdf_reader.search_pdf(**args)
            
            elif func_name == "list_pdfs":
                if not self.pdf_reader:
                    return {"error": "PDFReaderTool not available"}
                return self.pdf_reader.list_pdfs(**args)
            
            return {"error": f"Unknown tool: {func_name}"}

        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {"error": str(e)}

    # Legacy method support
    def parse_request(self, text: str) -> Optional[ToolRequest]:
        if not text: return None
        match = TOOL_REQUEST_PATTERN.search(text)
        if match:
            query = match.group("query").strip()
            return ToolRequest(name="tavily_search", args={"query": query})
        return None

    def execute(self, request: ToolRequest) -> ToolResult:
        result = self.route(request.name, request.args)
        formatted = str(result.get("result") or result)
        return ToolResult(name=request.name, args=request.args, output=result, formatted=formatted)
