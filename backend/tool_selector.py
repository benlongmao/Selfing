#!/usr/bin/env python3
"""
工具动态选择器 (Tool Selector)
优化Token消耗：精简描述 + 按需加载

[2026-01-20] 初始版本
- 将工具定义Token从~4800降低到~1500（普通对话）
- 核心工具常驻，扩展工具按意图动态加载

[2026-01-23] 方案A实施：语义理解
- 添加语义匹配替代纯关键词匹配
- 使用sentence-transformers计算语义相似度
- 混合策略：关键词（高置信）+ 语义（中置信）
"""

import logging
from typing import List, Dict, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Tool tiers / groups
# ============================================================

# [2026-03-20] Layer 0: always-on minimal toolkit + introspection
# [2026-04-06] rename_file lives in basic_file; agent_memory_* merged; recall_memory added
CORE_TOOLS = {
    "list_my_tools",
    "request_tool_group",
    "read_file",
    "write_file",
    "list_files",
    "recall_memory",
    "tavily_search",
    "get_current_time",
}

# Tier 2: intent-loaded tool groups
# [P3] Baseline groups supersede old monolithic "core"
TOOL_GROUPS = {
    # [2026-02-24] Introspection: "what tools / capabilities"
    "self_introspection": {
        "keywords": ["工具", "能力", "功能", "你能做什么", "你会什么", "你有什么", 
                     "tools", "capabilities", "what can you do", "abilities",
                     "我的工具", "我的能力", "列出工具", "查看工具"],
        "tools": ["list_my_tools"]
    },
    # [P3] Basic filesystem ops
    "basic_file": {
        "keywords": ["文件", "读", "写", "打开", "保存", "内容", "查看", "创建", "日记",
                     "重命名", "改名", "换文件名", "移动文件",
                     "file", "read", "write", "open", "save", "content", "create",
                     "rename", "mv", "move file",
                     "列出", "目录", "list", "dir", "ls", "看看", "有什么文件"],
        "tools": [
            "read_file",
            "write_file",
            "rename_file",
            "list_files",
            "search_files",
            "copy_file",
            "create_directory",
        ]
    },
    # [P3] Code execution
    "code_execution": {
        "keywords": ["运行", "执行", "代码", "python", "脚本", "计算", "程序",
                     "run", "execute", "code", "script", "calculate", "compute",
                     "测试", "试试", "帮我算", "写个"],
        "tools": ["execute_python"]
    },
    # [P3] Web search
    "web_search": {
        "keywords": ["搜索", "查询", "查找", "网上", "最新", "新闻", "了解",
                     "search", "query", "find", "web", "news", "latest", "look up",
                     "是什么", "怎么样", "什么是", "告诉我关于"],
        "tools": ["tavily_search"]
    },
    # [P3] Time utilities
    "time_util": {
        "keywords": ["时间", "几点", "日期", "今天", "现在", "time", "date", "now", "today", "when"],
        "tools": ["get_current_time"]
    },
    "agent_memory": {
        "keywords": [
            "状态快照",
            "运行记录",
            "落盘",
            "复核记录",
            "agent_memory",
            "外存记忆",
            "同步快照",
            "记录本轮",
        ],
        "tools": ["agent_memory_record", "agent_memory_sync"],
    },
    # [2026-03-13] Geo + weather
    "geo_weather": {
        "keywords": [
            "天气", "气温", "温度", "下雨", "降雨", "下雪", "晴天", "阴天", "多云", "风速", "湿度", "预报",
            "weather", "temperature", "rain", "snow", "sunny", "cloudy", "humidity", "forecast",
            "定位", "位置", "在哪", "在哪里", "我在哪", "ip位置", "城市", "经纬度",
            "location", "where am i", "my location", "city", "latitude", "longitude",
        ],
        "tools": ["get_location", "get_weather"]
    },
    # [P3] Goals (basic CRUD)
    "goal_basic": {
        "keywords": ["目标", "我的目标", "goal", "goals", "进度", "计划", "删除目标", "移除目标"],
        "tools": ["goal_list", "goal_add", "goal_detail", "goal_delete"]
    },
    "file_management": {
        "keywords": ["删除", "移动", "归档", "重复", "整理", "健康", "状态", "熵", "混乱", "乱",
                     "重命名", "改名",
                     "杂乱", "凌乱", "文件管理", "清理", "工作空间",
                     "delete", "move", "rename", "archive", "duplicate", "health", "entropy", "organize", "messy", "cleanup"],
        "tools": ["delete_file", "delete_directory", "rename_file", "batch_move_files", "detect_duplicate_files", 
                  "archive_by_date", "remove_duplicates", "analyze_workspace", "get_workspace_health"]
    },
    "code_analysis": {
        # [P3] Drop generic code/python tokens; keep analysis-specific keywords
        "keywords": ["分析代码", "代码质量", "依赖", "compare", "analyze_python", "质量检查", 
                     "函数分析", "类分析", "import分析", "沙盒", "代码结构", "静态分析"],
        "tools": ["analyze_python_file", "check_code_quality", "analyze_dependencies", "compare_python_files",
                  "code_sandbox_list", "code_sandbox_read"]
    },
    "calendar": {
        "keywords": ["日历", "日程", "事件", "提醒", "calendar", "event", "schedule", "时间线"],
        "tools": ["add_calendar_event", "list_calendar_events", "delete_calendar_event", 
                  "calculate_time_delta", "get_timeline_narrative", "get_time_context"]
    },
    "goal_management": {
        # [P3] Advanced goal ops; basics handled by goal_basic
        "keywords": ["里程碑", "milestone", "更新目标", "目标状态", "goal_update", "记录进度",
                     "删掉目标", "永久删除目标", "goal_delete"],
        "tools": ["goal_update_status", "goal_update_milestone", "goal_log_progress", "goal_delete"]
    },
    "email": {
        "keywords": [
            "邮件", "发送", "email", "mail", "发邮件", "收件箱", "检查邮件", "新邮件", "未读",
            "已发送", "发件箱",
        ],
        "tools": ["send_email", "check_unread_emails"]
    },
    "chemistry": {
        "keywords": ["化学", "分子", "SMILES", "化合物", "drug", "molecule", "rdkit", "化学式"],
        "tools": ["validate_smiles", "calculate_molecular_properties", "batch_validate_smiles",
                  "compare_molecules", "substructure_search"]
    },
    "stock": {
        "keywords": ["股票", "投资", "A股", "美股", "港股", "stock", "invest", "财务", "K线"],
        "tools": ["get_stock_info", "search_stocks", "technical_analysis", "evaluate_financial_health"]
    },
    "research": {
        "keywords": ["研究", "暂停研究", "恢复研究", "research"],
        "tools": ["research_pause", "research_resume", "research_status"]
    },
    "approval": {
        "keywords": ["审批", "批准", "approval", "pending"],
        "tools": ["approval_list_pending", "approval_check_status"]
    },
    "schedule": {
        "keywords": ["定时", "调度", "scheduled", "任务调度", "定时任务", "定时提醒", "系统提醒", "schedule_task", "schedule_list", "schedule_cancel", "删除定时", "列出定时", "取消定时"],
        "tools": ["schedule_task", "schedule_list", "schedule_cancel"]
    },
    "self_awareness": {
        "keywords": ["自我", "意识", "内省", "神游", "节律", "self", "inspect", "mind_wandering",
                     "我想做什么", "此刻", "想做什么",
                     # [2026-03-18] Self-study: architecture / config / implementation
                     "架构", "身体", "配置", "config", "settings", "源码", "源代码", "我的实现",
                     "如何运作", "机制", "我的代码", "backend", "docs", "研究自己"],
        "tools": ["get_self_facts", "inspect_self_code",
                  "read_self_code", "list_self_files", "search_self_code",
                  "request_mind_wandering", "set_my_rhythm"]
    },
    # [2026-02-07] Learning / planning / proposals
    "learning": {
        "keywords": ["学习", "知识", "learn", "knowledge", "记住", "学到", "理解", "掌握", "删除知识", "删掉知识", "修正知识", "更新知识", "记录下来", "值得记", "总结一下", "我发现了", "新发现", "结论是"],
        "tools": ["set_learning_goal", "add_learned_knowledge", "search_my_knowledge",
                  "update_my_knowledge", "delete_my_knowledge", "get_my_learning_goals", "get_knowledge_stats"]
    },
    "task_planning": {
        # [P3] Specific task-planning phrases
        # [2026-03-12] Full task_planning surface + continuous execution cues
        "keywords": ["执行计划", "分解任务", "任务规划", "步骤", "分解", "规划", "下一步", 
                     "execution_plan", "task_planning", "breakdown",
                     "创建任务", "列出任务", "我的任务", "今日计划", "待办", "todo",
                     "任务", "计划", "start_task", "complete_task", "持续任务执行",
                     "删除子任务", "删掉任务", "去掉一步", "remove task",
                     # [2026-03-30] Stale-task / resume nudges
                     "任务续跑", "任务续跑提醒", "尚未标记完成", "fail_task", "标记完成"],
        "tools": ["create_execution_plan", "add_task_to_plan", "delete_plan_task", "get_plan_details",
                  "get_next_task", "start_task", "complete_task", "fail_task", "get_my_active_plans",
                  "create_task", "list_tasks", "decompose_task", "plan_today"]
    },
    "code_proposal": {
        # [P3] Code-proposal specific keywords
        "keywords": ["代码提案", "修改提案", "改进提案", "优化代码", "propose_code", 
                     "code_change", "我的代码", "自我修改"],
        "tools": ["propose_code_change", "list_my_proposals", "check_proposal_result",
                  "read_my_code", "list_my_code_files", "analyze_code_structure"]
    },
    "companion": {
        "keywords": ["情感", "安全", "情绪", "emotion", "safety", "陪伴"],
        "tools": ["recall_memory", "analyze_user_emotion", "check_safety_risk"]
    },
    "browser": {
        "keywords": ["浏览器", "网页", "截图", "browser", "navigate", "screenshot", "webpage", "打开网站", "访问"],
        "tools": ["browse_web", "screenshot_page", "search_on_baidu", "get_browser_status"]
    },
    "browser_interaction": {
        "keywords": ["登录", "login", "发帖", "post", "填写", "表单", "form", "点击", "click", "输入", "提交", "submit", "注册", "register", "账号", "密码"],
        "tools": ["browse_web", "browser_fill_input", "browser_click", "browser_wait_element", 
                  "browser_get_text", "browser_get_elements", "browser_login", "browser_type"]
    },
    "moltbook": {
        "keywords": ["moltbook", "发帖", "评论", "点赞", "社交", "feed", "upvote", "community", "submolt", "删除", "delete", "关注", "follow", "用户", "profile", "创建社区"],
        "tools": ["moltbook_post", "moltbook_comment", "moltbook_get_comments",
                  "moltbook_upvote_post", "moltbook_upvote_comment", 
                  "moltbook_feed", "moltbook_search", "moltbook_get_post",
                  "moltbook_delete_post", "moltbook_list_communities", "moltbook_join_community",
                  "moltbook_leave_community", "moltbook_create_community",
                  "moltbook_get_my_profile", "moltbook_get_profile",
                  "moltbook_follow_user", "moltbook_unfollow_user"]
    },
    # [2026-02-22] Deep research / recall / viz / math
    "deep_research": {
        "keywords": ["研究", "深度研究", "调研", "研究报告", "多角度", "分析报告", "深入了解",
                     "research", "deep research", "investigation", "report", "综合分析"],
        "tools": ["deep_research", "list_research_reports"]
    },
    "memory_recall": {
        "keywords": ["记忆", "回忆", "之前", "以前", "上次", "历史", "我说过", "记得",
                     "memory", "recall", "remember", "history", "past", "previous",
                     "日记", "对话历史", "知识库"],
        "tools": ["recall_memory", "get_recent_context"]
    },
    "visualization": {
        "keywords": ["图表", "可视化", "画图", "折线图", "柱状图", "饼图", "散点图",
                     "chart", "plot", "graph", "visualize", "visualization",
                     "数据展示", "统计图"],
        "tools": ["create_chart"]
    },
    "data_analysis": {
        "keywords": ["统计", "分析数据", "平均值", "标准差", "中位数", "求解", "方程",
                     "导数", "积分", "数学", "statistics", "analyze", "mean", "median",
                     "derivative", "integral", "equation", "solve"],
        "tools": ["analyze_data", "solve_math"]
    },
    "browser_unified": {
        "keywords": ["浏览器", "网页", "截图", "browser", "navigate", "screenshot", 
                     "webpage", "打开网站", "访问", "交互", "点击", "填写"],
        "tools": ["browse_web", "screenshot_page", "browser_interact", "browser_login"]
    },
    # [2026-02-22] PDF + fetch
    "pdf_reader": {
        "keywords": ["pdf", "PDF", "论文", "报告", "文档", "paper", "document", 
                     "读pdf", "打开pdf", "看pdf", "pdf文件", "研究报告", "技术文档",
                     "arxiv", "下载pdf", "在线pdf", "url"],
        "tools": ["fetch_url_to_workspace", "read_pdf", "get_pdf_info", "search_pdf", "list_pdfs"]
    },
    "s_repo_evolution": {
        "keywords": [
            "改后端", "改代码", "改配置", "S项目", "本仓库", "仓库根", "项目根",
            "提交代码", "git diff", "git status", "pytest", "npm test", "npm run",
            "演进", "自举", "重构", "回滚", "checkout", "打标签", "里程碑",
            "backend/", "config/", "tool_router", "拉分支", "修bug", "跑测试",
            "evolution_", "execute_bash_project",
        ],
        "tools": [
            "evolution_read_file",
            "evolution_write_file",
            "evolution_mkdir",
            "evolution_delete_file",
            "evolution_delete_directory",
            "evolution_rename_path",
            "evolution_copy_path",
            "evolution_list_tree",
            "evolution_search_repo",
            "evolution_git_status",
            "evolution_git_diff",
            "evolution_git_log",
            "evolution_git_add",
            "evolution_git_commit",
            "evolution_git_checkout_file",
            "evolution_git_tag",
            "execute_bash_project",
        ],
    },
}


def agent_evolution_enabled() -> bool:
    """settings agent_evolution.enabled：注册演进工具且动态选择须始终并入 evolution_* 组。"""
    try:
        from backend.config import config

        ev = config.get("agent_evolution") or {}
        return isinstance(ev, dict) and bool(ev.get("enabled"))
    except Exception:
        return False


# Mirror TOOL_GROUPS["s_repo_evolution"] (only if ToolRouter registered those tools)
EVOLUTION_TOOL_NAMES: Set[str] = set(TOOL_GROUPS["s_repo_evolution"]["tools"])

# With agent_evolution: CORE + evolution_* + intent groups; trim cap > default 14
MAX_TOOLS_WITH_EVOLUTION = 40


# ============================================================
# Short tool blurbs (token savings)
# ============================================================

COMPACT_DESCRIPTIONS = {
    # Layer 0 meta
    "list_my_tools": "查看当前可用工具列表和可按需加载的工具组",
    "request_tool_group": "按需加载一组工具（如 email、calendar、browser 等）",
    # Core RW / search
    "read_file": "读取文件内容",
    "write_file": "创建或覆盖文件",
    "rename_file": "重命名或移动工作空间内单个文件",
    "copy_file": "在工作空间内复制单个文件",
    "create_directory": "在工作空间内创建目录（多级）",
    "list_files": "列出工作空间文件",
    "search_files": "搜索文件内容",
    "execute_python": "执行Python代码",
    "tavily_search": "网络搜索（新闻、实时信息）",
    "get_current_time": "获取当前时间",
    "get_location": "通过公网IP获取当前大致位置（城市/省份/国家/经纬度）",
    "get_weather": "查询天气：实时温度/湿度/风速/天气状况+近12小时预报，支持城市名或经纬度，不填则自动定位",
    "goal_list": "列出所有目标",
    
    # [2026-02-07] Learning + plans + proposals
    "set_learning_goal": "设定学习目标",
    "add_learned_knowledge": "记录学到的知识",
    "search_my_knowledge": "搜索知识库",
    "update_my_knowledge": "更新或修正知识库条目（标题/内容/置信度/标签）",
    "delete_my_knowledge": "从知识库删除一条知识（需 knowledge_id）",
    "create_execution_plan": "创建任务执行计划",
    "add_task_to_plan": "添加任务到计划",
    "delete_plan_task": "删除执行计划中的单条子任务，并修正计数与依赖",
    "get_next_task": "获取下一个任务",
    "create_task": "创建新任务。必填title，可选description/estimated_minutes/priority/deadline。用于记录待办。",
    "list_tasks": "列出任务。可选status(pending/in_progress/completed等)、limit。用于查看待办/进行中/已完成。",
    "decompose_task": "智能分解复杂任务为子任务。必填task_description。返回子任务列表和预估时间。",
    "plan_today": "制定今日计划。可选available_hours(默认8)。从待办中选任务并安排时间。",
    "propose_code_change": "提出代码修改建议",
    "read_my_code": "阅读我的代码",
    
    # Workspace hygiene
    "delete_file": "删除文件",
    "delete_directory": "删除目录",
    "batch_move_files": "批量移动文件",
    "detect_duplicate_files": "检测重复文件",
    "archive_by_date": "按日期归档文件",
    "remove_duplicates": "删除重复文件",
    "analyze_workspace": "分析工作空间",
    "get_workspace_health": "检查工作空间健康状态（熵值/建议）",
    
    # Static analysis
    "analyze_python_file": "分析Python文件结构",
    "check_code_quality": "检查代码质量",
    "analyze_dependencies": "分析依赖关系",
    "compare_python_files": "对比Python文件",
    "code_sandbox_list": "列出沙盒文件",
    "code_sandbox_read": "读取沙盒文件",
    
    # Calendar
    "add_calendar_event": "添加日历事件",
    "list_calendar_events": "列出日历事件",
    "delete_calendar_event": "删除日历事件",
    "calculate_time_delta": "计算时间差",
    "get_timeline_narrative": "获取时间线叙事",
    "get_time_context": "获取时间上下文",
    
    # Goals
    "goal_add": "添加新目标",
    "goal_update_status": "更新目标状态",
    "goal_update_milestone": "更新里程碑",
    "goal_log_progress": "记录进度",
    "goal_detail": "获取目标详情",
    "goal_delete": "永久删除目标及其里程碑与进度（仅本会话）",
    
    # Email
    "send_email": "发送邮件（支持附件）",
    "check_unread_emails": "查看收件箱或已发送(folder=sent)；limit 默认 5；only_unread 仅未读",
    
    # Chemistry
    "validate_smiles": "验证SMILES化学结构",
    "calculate_molecular_properties": "计算分子属性（MW/LogP/TPSA）",
    "batch_validate_smiles": "批量验证SMILES",
    "compare_molecules": "计算分子相似度",
    "substructure_search": "子结构搜索",
    
    # Equities
    "get_stock_info": "获取股票信息（CN/US/HK）",
    "search_stocks": "搜索股票",
    "technical_analysis": "技术分析（K线指标）",
    "evaluate_financial_health": "评估财务健康度",
    
    # Research engine
    "research_pause": "暂停研究",
    "research_resume": "恢复研究",
    "research_status": "查看研究状态",
    
    # Approvals
    "approval_list_pending": "列出待审批项",
    "approval_check_status": "检查审批状态",
    
    # Scheduler
    "schedule_task": "创建定时任务",
    "schedule_list": "列出定时任务",
    "schedule_cancel": "取消定时任务",
    
    # Self-model / code introspection
    "get_self_facts": "获取自我认知事实",
    "inspect_self_code": "查看workspace内文件",
    "read_self_code": "读取自身源码(backend/config/docs)",
    "list_self_files": "列出自身目录结构",
    "search_self_code": "在自身代码中搜索关键词",
    "request_mind_wandering": "请求神游",
    "set_my_rhythm": "设置自主节律",
    
    # Companion (shares recall_memory with memory_recall; avoid recall_memories)
    "recall_memory": "综合回忆（对话/日记/知识/规则）",
    "analyze_user_emotion": "分析用户情感",
    "check_safety_risk": "检查安全风险",
    
    # Browser basics
    "browse_web": "访问网页",
    "screenshot_page": "截图网页",
    "search_on_baidu": "百度搜索",
    "get_browser_status": "浏览器状态",
    
    # Browser interaction (login, forms, etc.)
    "browser_fill_input": "填写表单",
    "browser_click": "点击元素",
    "browser_wait_element": "等待元素",
    "browser_get_text": "获取文本",
    "browser_get_elements": "获取元素列表",
    "browser_login": "一键登录",
    "browser_type": "模拟输入",
    
    # [removed] get_tavily_stats

    # Moltbook
    "moltbook_post": "Moltbook发帖（⏳30分钟冷却）",
    "moltbook_comment": "Moltbook评论（⏳20秒冷却，每日50条上限）",
    "moltbook_get_comments": "获取帖子评论列表",
    "moltbook_upvote_post": "点赞帖子",
    "moltbook_upvote_comment": "点赞评论",
    "moltbook_feed": "查看Moltbook Feed",
    "moltbook_search": "Moltbook AI语义搜索",
    "moltbook_get_post": "获取帖子详情",
    "moltbook_delete_post": "删除自己的帖子（⚠️不可撤销）",
    "moltbook_list_communities": "浏览Moltbook社区",
    "moltbook_join_community": "加入Moltbook社区",
    "moltbook_leave_community": "退出Moltbook社区",
    "moltbook_create_community": "创建新的Moltbook社区",
    "moltbook_get_my_profile": "获取自己的用户信息",
    "moltbook_get_profile": "获取其他用户信息",
    "moltbook_follow_user": "关注用户（⚠️要谨慎）",
    "moltbook_unfollow_user": "取消关注用户",
    
    # [2026-02-22] Research / charts / math
    "deep_research": "深度研究（多源搜索+报告生成）",
    "list_research_reports": "列出研究报告",
    "get_recent_context": "获取最近记忆上下文",
    "create_chart": "创建图表（折线/柱状/饼图）",
    "analyze_data": "数据统计分析",
    "solve_math": "数学求解（方程/导数/积分）",
    "browser_interact": "浏览器交互（点击/填写/获取）",
    
    # [2026-02-22] PDF
    "fetch_url_to_workspace": "https 下载到工作区（服务端），再 read_pdf",
    "read_pdf": "读取PDF内容（按页提取文本）",
    "get_pdf_info": "获取PDF信息（页数/作者/标题）",
    "search_pdf": "在PDF中搜索关键词",
    "list_pdfs": "列出目录中的PDF文件",

    "agent_memory_record": "写入 agent_memory 运行记录 JSON",
    "agent_memory_sync": "刷新 agent_memory 状态快照 Markdown",

    # [2026-04-03] Repo evolution (agent_evolution)
    "evolution_read_file": "读 S 仓库内文件（相对根路径）",
    "evolution_write_file": "写 S 仓库内文件（禁止 .env / 勿直写 .git）",
    "evolution_mkdir": "在 S 仓库内创建目录",
    "evolution_delete_file": "删除 S 仓库内文件",
    "evolution_delete_directory": "删除 S 仓库内目录（可 recursive）",
    "evolution_rename_path": "重命名或移动 S 仓库内路径",
    "evolution_copy_path": "复制 S 仓库内文件或目录",
    "evolution_list_tree": "列出 S 仓库子树（文本）",
    "evolution_search_repo": "在 S 仓库内按关键词搜源码/配置",
    "evolution_git_status": "git status（porcelain）",
    "evolution_git_diff": "git diff（可选 staged）",
    "evolution_git_log": "git log oneline",
    "evolution_git_add": "git add 路径或 add -u",
    "evolution_git_commit": "git commit（需说明）",
    "evolution_git_checkout_file": "从某 revision 恢复单个文件",
    "evolution_git_tag": "打 git 标签",
    "execute_bash_project": "在 S 仓库根执行受限 bash（含 pytest、python -m compileall、npm test|run|ci）",
}


class ToolSelector:
    """动态工具选择器"""
    
    def __init__(self, tool_router):
        """
        初始化工具选择器
        
        Args:
            tool_router: ToolRouter实例，用于获取完整工具定义
        """
        self.tool_router = tool_router
        self._full_definitions_cache: Optional[List[Dict]] = None
        self._tool_name_to_def: Dict[str, Dict] = {}
        
    def _build_cache(self):
        """构建工具定义缓存"""
        if self._full_definitions_cache is None:
            self._full_definitions_cache = self.tool_router.get_tool_definitions()
            for tool_def in self._full_definitions_cache:
                name = tool_def.get("function", {}).get("name")
                if name:
                    self._tool_name_to_def[name] = tool_def
            logger.info(f"[ToolSelector] Built cache with {len(self._tool_name_to_def)} tools")
    
    def _detect_needed_groups(self, user_input: str) -> Set[str]:
        """
        检测用户输入需要哪些工具组
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            需要的工具组名称集合
        """
        needed_groups = set()
        lower_input = user_input.lower()
        
        for group_name, group_info in TOOL_GROUPS.items():
            keywords = group_info["keywords"]
            for keyword in keywords:
                if keyword.lower() in lower_input or keyword in user_input:
                    needed_groups.add(group_name)
                    break
        
        return needed_groups
    
    def _get_tool_def_compact(self, tool_name: str) -> Optional[Dict]:
        """
        获取精简版工具定义
        
        Args:
            tool_name: 工具名称
            
        Returns:
            精简版工具定义，如果工具不存在返回None
        """
        self._build_cache()
        
        original_def = self._tool_name_to_def.get(tool_name)
        if not original_def:
            return None
        
        # Clone OpenAI-shaped def, swap description
        compact_def = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": COMPACT_DESCRIPTIONS.get(tool_name, 
                    original_def.get("function", {}).get("description", "")),
                "parameters": original_def.get("function", {}).get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
        }
        
        return compact_def
    
    MAX_TOOLS_PER_TURN = 14
    
    HIGH_PRIORITY_GROUPS = {
        "basic_file",
        "code_execution",
        "web_search",
        "time_util",
        "goal_basic",
        "schedule",
        "s_repo_evolution",
    }
    
    # Layer 0 tools never trimmed away
    ESSENTIAL_TOOLS = set(CORE_TOOLS)

    def _dynamic_max_tools(self) -> int:
        return MAX_TOOLS_WITH_EVOLUTION if agent_evolution_enabled() else self.MAX_TOOLS_PER_TURN

    def _essential_for_trim(self) -> Set[str]:
        e = set(self.ESSENTIAL_TOOLS)
        if agent_evolution_enabled():
            e |= EVOLUTION_TOOL_NAMES
        return e
    
    def select_tools(self, user_input: str, use_compact: bool = True) -> List[Dict]:
        """
        根据用户输入动态选择工具
        
        Args:
            user_input: 用户输入文本
            use_compact: 是否使用精简描述
            
        Returns:
            选中的工具定义列表
        """
        self._build_cache()
        
        # 1. Start from CORE (+ evolution tools when enabled)
        selected_tools: Set[str] = set(CORE_TOOLS)
        if agent_evolution_enabled():
            selected_tools.update(EVOLUTION_TOOL_NAMES)
        
        # 2. Keyword-detect tool groups
        needed_groups = self._detect_needed_groups(user_input)
        
        # [P3] Load HIGH_PRIORITY_GROUPS first
        high_priority = [g for g in needed_groups if g in self.HIGH_PRIORITY_GROUPS]
        low_priority = [g for g in needed_groups if g not in self.HIGH_PRIORITY_GROUPS]
        sorted_groups = high_priority + low_priority
        
        for group_name in sorted_groups:
            group_tools = TOOL_GROUPS.get(group_name, {}).get("tools", [])
            selected_tools.update(group_tools)
            logger.debug(f"[ToolSelector] Added group '{group_name}': {group_tools}")
        
        # [P3] Empty selection → basic file quartet
        if not selected_tools:
            default_tools = ["read_file", "write_file", "rename_file", "list_files"]
            selected_tools.update(default_tools)
            logger.debug(f"[ToolSelector] No match, added defaults: {default_tools}")
        
        # [P3] Enforce per-turn cap with essential retention
        _cap = self._dynamic_max_tools()
        _ess = self._essential_for_trim()
        if len(selected_tools) > _cap:
            essential = selected_tools & _ess
            others = list(selected_tools - essential)
            remaining_slots = _cap - len(essential)
            selected_tools = essential | set(others[: max(0, remaining_slots)])
            logger.debug(f"[ToolSelector] Trimmed to {len(selected_tools)} tools (max={_cap})")
        
        # 3. Materialize defs
        result = []
        for tool_name in selected_tools:
            if use_compact:
                tool_def = self._get_tool_def_compact(tool_name)
            else:
                tool_def = self._tool_name_to_def.get(tool_name)
            
            if tool_def:
                result.append(tool_def)
        
        # 4. Drop defs missing from router (optional deps)
        available_names = set(self._tool_name_to_def.keys())
        result = [t for t in result if t["function"]["name"] in available_names]
        
        logger.info(f"[ToolSelector] Selected {len(result)} tools "
                   f"(groups: {sorted_groups or ['defaults']}) "
                   f"for input: {user_input[:50]}...")
        
        return result
    
    def get_all_tools_compact(self) -> List[Dict]:
        """
        获取所有工具的精简版定义
        
        Returns:
            所有工具的精简版定义列表
        """
        self._build_cache()
        
        result = []
        for tool_name in self._tool_name_to_def.keys():
            tool_def = self._get_tool_def_compact(tool_name)
            if tool_def:
                result.append(tool_def)
        
        return result


# Module singleton
_selector_instance: Optional[ToolSelector] = None


def get_tool_selector(tool_router) -> ToolSelector:
    """获取工具选择器单例"""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = ToolSelector(tool_router)
    return _selector_instance


def select_tools_for_input(tool_router, user_input: str, use_compact: bool = True) -> List[Dict]:
    """
    便捷函数：根据用户输入选择工具
    
    Args:
        tool_router: ToolRouter实例
        user_input: 用户输入
        use_compact: 是否使用精简描述
        
    Returns:
        选中的工具定义列表
    """
    selector = get_tool_selector(tool_router)
    return selector.select_tools(user_input, use_compact)


# [2026-04-02] Three-layer gate: NEED markers + TOOL_GROUPS cross-check + semantic fallback
# Bias toward false positives (extra tools) over missing a needed tool

NO_TOOL_MARKERS = [
    "你觉得", "你认为", "你的看法", "你怎么看",
    "谢谢", "好的", "明白了", "懂了", "知道了", "嗯", "哦",
    "你好", "早上好", "晚上好", "你是谁",
]
NEED_TOOL_MARKERS = [
    "帮我", "请", "执行", "运行", "创建", "建立", "新建", "生成",
    "删除", "移动", "整理", "复制", "拷贝",
    "读取", "写入", "搜索", "查找", "分析", "检查", "测试", "看看", "看一下",
    "修改", "编辑", "更新", "保存", "下载", "上传",
]

_TOOL_GATE_SEMANTIC: Optional[Dict[str, np.ndarray]] = None


def _semantic_tool_gate(text: str, threshold: float = 0.48) -> bool:
    """语义兜底：被关键词判为纯聊天时，用 embedder 做最后一道校验。"""
    global _TOOL_GATE_SEMANTIC
    if _TOOL_GATE_SEMANTIC is None:
        try:
            from backend.embedder import get_embedder
            from backend.tool_semantic_descriptions import TOOL_GROUPS_SEMANTIC
            enc = get_embedder()
            _TOOL_GATE_SEMANTIC = {}
            for gname, ginfo in TOOL_GROUPS_SEMANTIC.items():
                combined = ginfo["description"] + "\n" + "\n".join(ginfo["examples"])
                _TOOL_GATE_SEMANTIC[gname] = enc.encode(combined[:512], normalize=True)
        except Exception:
            _TOOL_GATE_SEMANTIC = {}
            return False
    if not _TOOL_GATE_SEMANTIC:
        return False
    try:
        from backend.embedder import get_embedder
        qv = get_embedder().encode(text[:256], normalize=True)
        return any(float(np.dot(qv, rv)) >= threshold for rv in _TOOL_GATE_SEMANTIC.values())
    except Exception:
        return False


def should_provide_full_tools(user_input: str) -> bool:
    """
    判定本轮是否应做完整的动态工具选择（Layer 0 + Layer 1）。
    返回 False 时仅下发 Layer 0 永驻工具。

    三层策略：
    1. NEED_TOOL_MARKERS 快速放行
    2. TOOL_GROUPS 关键词交叉检查（防止 NO_TOOL_MARKERS 误杀）
    3. 语义兜底（处理关键词全部 miss 的边缘 case）
    """
    if any(m in user_input for m in NEED_TOOL_MARKERS):
        return True

    lower_input = user_input.lower()
    for group_info in TOOL_GROUPS.values():
        if any(kw.lower() in lower_input for kw in group_info["keywords"]):
            return True

    is_pure_chat = len(user_input) < 60 and any(m in user_input for m in NO_TOOL_MARKERS)
    if not is_pure_chat:
        return True

    return _semantic_tool_gate(lower_input)


# Back-compat alias
should_provide_tools_for_input = should_provide_full_tools


# ============================================================
# [2026-01-23] Semantic tool-group matcher
# ============================================================

class SemanticToolSelector:
    """
    基于语义相似度的工具选择器
    使用sentence-transformers计算用户输入与工具组描述的语义匹配度
    """
    
    def __init__(self, embedder, tool_router):
        """
        初始化语义选择器
        
        Args:
            embedder: 嵌入模型实例（来自backend.embedder）
            tool_router: ToolRouter实例
        """
        self.embedder = embedder
        self.tool_router = tool_router
        self.group_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False
        
        logger.info("[SemanticToolSelector] Initializing...")
    
    def _initialize_embeddings(self):
        """预计算所有工具组的语义嵌入向量"""
        if self._initialized:
            return
        
        try:
            from backend.tool_semantic_descriptions import TOOL_GROUPS_SEMANTIC
            
            logger.info("[SemanticToolSelector] Computing embeddings for tool groups...")
            
            for group_name, group_info in TOOL_GROUPS_SEMANTIC.items():
                # Concat description + examples for embedding
                description = group_info["description"]
                examples = "\n".join(group_info["examples"])
                combined_text = f"{description}\n\n示例:\n{examples}"
                
                # embedder.encode(...)
                embedding = self.embedder.encode(combined_text)
                self.group_embeddings[group_name] = embedding
                
                logger.debug(f"[SemanticToolSelector] Embedded '{group_name}': {embedding.shape}")
            
            self._initialized = True
            logger.info(f"[SemanticToolSelector] Initialized {len(self.group_embeddings)} tool group embeddings")
            
        except Exception as e:
            logger.error(f"[SemanticToolSelector] Failed to initialize embeddings: {e}")
            # Fallback: keyword-only hybrid path
            self._initialized = False
    
    def compute_similarity(self, text1_embedding: np.ndarray, text2_embedding: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度
        
        Args:
            text1_embedding: 第一个文本的嵌入向量
            text2_embedding: 第二个文本的嵌入向量
            
        Returns:
            余弦相似度（0-1）
        """
        # Cosine similarity
        dot_product = np.dot(text1_embedding, text2_embedding)
        norm1 = np.linalg.norm(text1_embedding)
        norm2 = np.linalg.norm(text2_embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Map cosine [-1,1] → [0,1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def select_by_semantic(self, user_input: str, threshold: float = 0.5) -> Set[str]:
        """
        基于语义相似度选择工具组
        
        Args:
            user_input: 用户输入
            threshold: 相似度阈值（0-1），超过此值则激活工具组
            
        Returns:
            选中的工具组名称集合
        """
        if not self._initialized:
            self._initialize_embeddings()
        
        if not self._initialized:
            # Init failed → no semantic groups
            return set()
        
        # Encode user utterance
        user_embedding = self.embedder.encode(user_input)
        
        selected_groups = set()
        similarities = {}
        
        # Score each group embedding
        for group_name, group_embedding in self.group_embeddings.items():
            similarity = self.compute_similarity(user_embedding, group_embedding)
            similarities[group_name] = similarity
            
            if similarity >= threshold:
                selected_groups.add(group_name)
                logger.info(f"[Semantic] Activated '{group_name}' (similarity={similarity:.3f})")
            else:
                logger.debug(f"[Semantic] Rejected '{group_name}' (similarity={similarity:.3f} < {threshold})")
        
        # Soft fallback: best group if within 80% of threshold
        if not selected_groups and similarities:
            max_group = max(similarities.items(), key=lambda x: x[1])
            if max_group[1] >= threshold * 0.8:  # 20% relaxed floor
                selected_groups.add(max_group[0])
                logger.info(f"[Semantic] Fallback activated '{max_group[0]}' (similarity={max_group[1]:.3f})")
        
        return selected_groups


class HybridToolSelector(ToolSelector):
    """
    混合工具选择器：关键词 + 语义
    
    策略：
    1. 先尝试关键词匹配（快速、精确）
    2. 如果关键词未匹配，使用语义匹配（泛化、容错）
    3. 合并结果
    """
    
    def __init__(self, tool_router, embedder=None):
        """
        初始化混合选择器
        
        Args:
            tool_router: ToolRouter实例
            embedder: 可选的嵌入模型，如果不提供则尝试自动获取
        """
        super().__init__(tool_router)
        
        # Optional semantic sidecar
        self.semantic_selector = None
        if embedder:
            try:
                self.semantic_selector = SemanticToolSelector(embedder, tool_router)
                logger.info("[HybridToolSelector] Semantic matching enabled")
            except Exception as e:
                logger.warning(f"[HybridToolSelector] Failed to init semantic selector: {e}")
    
    def select_tools(self, user_input: str, use_compact: bool = True) -> List[Dict]:
        """
        混合策略选择工具
        
        Args:
            user_input: 用户输入文本
            use_compact: 是否使用精简描述
            
        Returns:
            选中的工具定义列表
        """
        self._build_cache()
        
        # 1. Always include CORE (+ evolution)
        selected_tools: Set[str] = set(CORE_TOOLS)
        if agent_evolution_enabled():
            selected_tools.update(EVOLUTION_TOOL_NAMES)
        
        # 2. Keyword groups (high precision)
        keyword_groups = self._detect_needed_groups(user_input)
        semantic_groups: Set[str] = set()
        
        if keyword_groups:
            logger.info(f"[Hybrid] Keyword matched: {keyword_groups}")
            for group_name in keyword_groups:
                group_tools = TOOL_GROUPS.get(group_name, {}).get("tools", [])
                selected_tools.update(group_tools)
        
        # 3. Semantic groups when keywords miss
        if not keyword_groups and self.semantic_selector:
            try:
                # Semantic threshold 0.42 (tuned offline)
                # Rough guide: 0.30-0.42 aggressive; 0.42-0.50 balanced; 0.50+ conservative
                semantic_groups = self.semantic_selector.select_by_semantic(user_input, threshold=0.42)
                
                if semantic_groups:
                    logger.info(f"[Hybrid] Semantic matched: {semantic_groups}")
                    for group_name in semantic_groups:
                        group_tools = TOOL_GROUPS.get(group_name, {}).get("tools", [])
                        selected_tools.update(group_tools)
                else:
                    logger.debug(f"[Hybrid] No semantic match for: {user_input[:50]}...")
            except Exception as e:
                logger.warning(f"[Hybrid] Semantic matching failed: {e}")
        
        # [2026-02-24] Still empty → seed defaults + list_my_tools
        if not selected_tools:
            default_tools = ["read_file", "write_file", "rename_file", "list_files", "list_my_tools"]
            selected_tools.update(default_tools)
            logger.debug(f"[Hybrid] No match, added defaults: {default_tools}")
        
        # [2026-03-13] Same trim policy as base: keep ESSENTIALS, then HIGH_PRIORITY order
        # [2026-04-03] Higher cap + evolution_* always in essential set when enabled
        matched_groups = keyword_groups | semantic_groups
        _cap = self._dynamic_max_tools()
        _ess = self._essential_for_trim()
        if len(selected_tools) > _cap:
            essential = selected_tools & _ess
            remaining_slots = _cap - len(essential)
            if matched_groups:
                high_first = [g for g in self.HIGH_PRIORITY_GROUPS if g in matched_groups]
                rest_groups = [g for g in matched_groups if g not in self.HIGH_PRIORITY_GROUPS]
                sorted_groups = high_first + rest_groups
                ordered_tool_list: List[str] = []
                for g in sorted_groups:
                    for t in TOOL_GROUPS.get(g, {}).get("tools", []):
                        if t in selected_tools and t not in _ess and t not in ordered_tool_list:
                            ordered_tool_list.append(t)
                selected_tools = essential | set(ordered_tool_list[: max(0, remaining_slots)])
            else:
                others = list(selected_tools - essential)
                selected_tools = essential | set(others[: max(0, remaining_slots)])
            logger.debug(f"[Hybrid] Trimmed to {len(selected_tools)} tools (max={_cap})")
        
        # 4. Build def list
        result = []
        for tool_name in selected_tools:
            if use_compact:
                tool_def = self._get_tool_def_compact(tool_name)
            else:
                tool_def = self._tool_name_to_def.get(tool_name)
            
            if tool_def:
                result.append(tool_def)
        
        # 5. Filter unavailable tool names
        available_names = set(self._tool_name_to_def.keys())
        result = [t for t in result if t["function"]["name"] in available_names]
        
        strategy = "keyword" if keyword_groups else ("semantic" if not keyword_groups and self.semantic_selector else "core_only")
        logger.info(f"[Hybrid] Selected {len(result)} tools (strategy: {strategy}) for: {user_input[:50]}...")
        
        return result


# Hybrid selector singleton
_hybrid_selector_instance: Optional[HybridToolSelector] = None


def get_hybrid_tool_selector(tool_router, embedder=None) -> HybridToolSelector:
    """获取混合工具选择器单例"""
    global _hybrid_selector_instance
    if _hybrid_selector_instance is None:
        _hybrid_selector_instance = HybridToolSelector(tool_router, embedder)
    return _hybrid_selector_instance


def select_tools_with_semantic(tool_router, user_input: str, embedder=None, use_compact: bool = True) -> List[Dict]:
    """
    使用混合策略（关键词+语义）选择工具
    
    Args:
        tool_router: ToolRouter实例
        user_input: 用户输入
        embedder: 嵌入模型实例
        use_compact: 是否使用精简描述
        
    Returns:
        选中的工具定义列表
    """
    selector = get_hybrid_tool_selector(tool_router, embedder)
    return selector.select_tools(user_input, use_compact)
