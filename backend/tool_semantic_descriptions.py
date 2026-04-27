#!/usr/bin/env python3
"""
Semantic blurbs per tool group (descriptions + examples) for embedding-based tool selection.

[2026-01-23] Plan A: prefer semantic retrieval over raw keyword routing.
"""

# Each group: narrative description + example utterances (CN+EN mix intentional for recall)
TOOL_GROUPS_SEMANTIC = {
    "stock": {
        "description": """
        股票市场分析和投资决策支持工具组。
        
        核心功能：
        - 查询股票实时价格、历史行情数据
        - 技术指标分析（MA、MACD、RSI、布林带等）
        - 财务健康度评估（ROE、利润率、估值指标）
        - 股票搜索和筛选
        
        适用场景：
        - 用户询问某只股票的情况、价格、走势
        - 需要投资建议、买卖判断
        - 评估公司财务状况、盈利能力
        - 进行技术面或基本面分析
        - 了解市场行情、板块情况
        """,
        "examples": [
            "帮我看看平安银行怎么样",
            "分析一下苹果公司的股票",
            "我想了解茅台的财务状况", 
            "这只股票能不能买",
            "最近A股行情如何",
            "查询000001的价格",
            "帮我分析这家公司",
            "检查你的股票工具",
            "你能做投资分析吗",
            "有金融分析能力吗"
        ],
        "tools": ["get_stock_info", "search_stocks", "technical_analysis", "evaluate_financial_health"]
    },
    
    "chemistry": {
        "description": """
        化学分子结构分析和药物化学计算工具组。
        
        核心功能：
        - SMILES化学结构语法验证
        - 分子属性计算（分子量、LogP、TPSA、氢键等）
        - 分子相似度对比（Tanimoto系数）
        - 子结构搜索和匹配
        - 批量化合物验证
        
        适用场景：
        - 验证化学结构式的正确性
        - 计算药物分子的理化性质
        - 药物研发和化合物筛选
        - 分子设计和优化
        - 化学数据库查询
        """,
        "examples": [
            "验证这个SMILES结构是否正确",
            "计算这个分子的LogP",
            "这两个化合物相似吗",
            "查找苯环结构",
            "分析这个药物分子",
            "计算化合物的性质",
            "验证结构式",
            "药物研发工具",
            "化学计算功能"
        ],
        "tools": ["validate_smiles", "calculate_molecular_properties", "batch_validate_smiles",
                  "compare_molecules", "substructure_search"]
    },
    
    "browser": {
        "description": """
        网页浏览器自动化控制工具组。
        
        核心功能：
        - 导航到指定URL打开网页
        - 网页截图（全页或局部）
        - 百度搜索自动化
        - 获取网页内容和状态
        
        适用场景：
        - 需要打开和访问特定网站
        - 获取网页信息和内容
        - 自动化网页操作
        - 网页内容截图保存
        - 搜索引擎查询
        """,
        "examples": [
            "打开百度搜索Python教程",
            "帮我访问某个网页",
            "截图保存当前页面",
            "浏览这个网站",
            "打开一个URL",
            "网页截图",
            "查看网站内容",
            "访问页面",
            "打开网站"
        ],
        "tools": ["browse_web", "screenshot_page", "search_on_baidu", "get_browser_status"]
    },
    
    "browser_interaction": {
        "description": """
        网站交互工具组 - 登录、表单填写、发帖等高级浏览器操作。
        
        核心功能：
        - 网站登录（用户名、密码、验证）
        - 表单填写（输入框、文本域）
        - 元素点击（按钮、链接）
        - 内容提取（获取文本、元素列表）
        - 发帖、注册、提交表单
        
        适用场景：
        - 需要登录到某个网站
        - 在论坛、社区发帖
        - 填写和提交表单
        - 自动化网站操作
        - 账号注册
        - 需要输入密码或用户名
        """,
        "examples": [
            "登录到moltbook.com",
            "帮我在论坛发个帖子",
            "填写注册表单",
            "用账号密码登录",
            "点击提交按钮",
            "在网站上发布内容",
            "登录后台系统",
            "填写用户名和密码",
            "在这个网站注册",
            "提交表单",
            "发布一篇文章",
            "登录我的账号"
        ],
        "tools": ["browse_web", "browser_fill_input", "browser_click", "browser_wait_element", 
                  "browser_get_text", "browser_get_elements", "browser_login", "browser_type"]
    },
    
    "email": {
        "description": """
        电子邮件收发工具组。
        
        核心功能：
        - 发送邮件（支持附件）
        - 检查未读邮件
        - 邮件通知和提醒
        
        适用场景：
        - 需要发送消息、通知给用户
        - 发送文件、报告、分析结果
        - 检查是否有新邮件
        - 邮件联系和沟通
        """,
        "examples": [
            "给张三发个邮件",
            "发送这个文件给用户",
            "通知他这个消息",
            "给我发邮件提醒",
            "检查有没有新邮件",
            "检查收件箱有些什么邮件",
            "看看收件箱",
            "发消息给某人",
            "把这个发出去",
            "联系用户"
        ],
        "tools": ["send_email", "check_unread_emails"]
    },
    
    "file_management": {
        "description": """
        文件管理和整理工具组。
        
        核心功能：
        - 批量移动文件到目标目录
        - 检测和删除重复文件
        - 按日期自动归档文件
        - 工作空间分析和整理
        - 工作空间健康检查（熵值、建议）
        - 删除文件和目录
        
        适用场景：
        - 需要整理混乱的文件
        - 清理重复文件释放空间
        - 按时间归档历史文件
        - 批量文件操作
        - 检查工作空间状态和健康度
        """,
        "examples": [
            "帮我整理这些文件",
            "删除重复的文件",
            "把旧文件归档",
            "清理工作空间",
            "批量移动文件",
            "删除这个目录",
            "整理一下文件",
            "检查工作空间状态",
            "工作空间健康情况",
            "我的文件乱不乱",
            "重命名工作区里的文件",
        ],
        "tools": ["delete_file", "delete_directory", "rename_file", "batch_move_files", "detect_duplicate_files",
                  "archive_by_date", "remove_duplicates", "analyze_workspace", "get_workspace_health"]
    },
    
    "code_analysis": {
        "description": """
        代码分析和质量检查工具组。
        
        核心功能：
        - Python文件结构分析（函数、类、导入）
        - 代码质量检查（复杂度、风格）
        - 依赖关系分析
        - 代码文件对比
        - 沙盒代码查看
        
        适用场景：
        - 分析Python代码结构
        - 检查代码质量和规范
        - 了解代码依赖关系
        - 对比代码差异
        """,
        "examples": [
            "分析这个Python文件",
            "检查代码质量",
            "查看依赖关系",
            "对比两个文件",
            "代码结构分析",
            "质量检查"
        ],
        "tools": ["analyze_python_file", "check_code_quality", "analyze_dependencies", 
                  "compare_python_files", "code_sandbox_list", "code_sandbox_read"]
    },
    
    "calendar": {
        "description": """
        日历和事件管理工具组（每条为单次时间点，无周期间隔重复）。
        
        核心功能：
        - 添加单次日历事件或时间标记
        - 查询特定日期的事件
        - 删除事件
        - 计算时间间隔
        - 获取时间线叙事
        
        适用场景：
        - 记录「某一天某一刻」的安排或备忘
        - 查看历史事件、时间线
        - 时间计算和规划
        
        重要：需要按天/周/小时重复执行的请用定时任务组 schedule_task，勿仅用日历标题写「每周」。
        """,
        "examples": [
            "添加一个日程",
            "明天的安排是什么",
            "记录这个事件",
            "查看日历",
            "删除这个事件",
            "时间线"
        ],
        "tools": ["add_calendar_event", "list_calendar_events", "delete_calendar_event",
                  "calculate_time_delta", "get_timeline_narrative", "get_time_context"]
    },
    
    "goal_management": {
        "description": """
        目标和任务管理工具组。
        
        核心功能：
        - 创建和管理目标
        - 更新目标状态和进度
        - 设置里程碑
        - 记录完成情况
        - 查看目标详情
        
        适用场景：
        - 设定长期目标和计划
        - 跟踪项目进度
        - 管理任务清单
        - 里程碑管理
        """,
        "examples": [
            "创建一个新目标",
            "更新任务进度",
            "查看我的计划",
            "设置里程碑",
            "记录完成情况",
            "永久删除某个目标",
            "目标管理"
        ],
        "tools": ["goal_add", "goal_update_status", "goal_update_milestone", 
                  "goal_log_progress", "goal_detail", "goal_list", "goal_delete"]
    },
    
    "self_awareness": {
        "description": """
        自我认知和内省工具组。
        
        核心功能：
        - 获取自我状态硬事实（z_self维度、能量、漂移）
        - 查看自身代码和配置（read_self_code 可读 backend/、config/、docs/）
        - 列出和搜索自己的源代码结构
        - 更新意识配置
        - 请求神游（深度思考）
        - 设置自主节律
        
        适用场景：
        - Agent 自我检查和内省
        - 研究自身架构、了解自己的身体（S项目实现）
        - 做技术研究前查看 config/settings.yaml、docs/ 等
        - 了解当前状态和能力
        - 进入深度思考模式
        """,
        "examples": [
            "你现在的状态怎么样",
            "检查你自己",
            "查看你的能量",
            "进入神游模式",
            "自我检查",
            "内省一下",
            "你的意识状态",
            "研究我的架构",
            "看看 config 配置",
            "我的身体是怎么实现的",
            "查看 backend 的代码结构"
        ],
        "tools": ["get_self_facts", "inspect_self_code",
                  "read_self_code", "list_self_files", "search_self_code",
                  "request_mind_wandering", "set_my_rhythm"]
    },
    
    "companion": {
        "description": """
        伴侣和情感支持工具组。
        
        核心功能：
        - 回忆和检索记忆
        - 分析用户情感状态
        - 安全风险检查
        - 获取时间上下文
        
        适用场景：
        - 理解用户情绪和情感
        - 回忆历史对话和信息
        - 安全性判断
        - 情感陪伴
        """,
        "examples": [
            "你还记得我之前说的吗",
            "分析我的情绪",
            "这个话题安全吗",
            "理解我的感受",
            "回忆一下",
            "情感分析"
        ],
        "tools": ["recall_memory", "analyze_user_emotion", "check_safety_risk", "get_time_context"]
    },
    
    "research": {
        "description": """
        研究和探索引擎工具组。
        
        核心功能：
        - 暂停研究进程
        - 恢复研究进程  
        - 查看研究状态
        
        适用场景：
        - 控制后台研究任务
        - 管理长期探索项目
        """,
        "examples": [
            "暂停研究",
            "继续研究",
            "研究状态如何",
            "停止探索",
            "恢复学习"
        ],
        "tools": ["research_pause", "research_resume", "research_status"]
    },
    
    "approval": {
        "description": """
        审批和确认工具组。
        
        核心功能：
        - 查看待审批事项
        - 检查审批状态
        
        适用场景：
        - 查看需要确认的操作
        - 检查审批进度
        """,
        "examples": [
            "有什么需要我审批的",
            "查看待确认事项",
            "审批状态",
            "需要我批准的",
            "等待处理的"
        ],
        "tools": ["approval_list_pending", "approval_check_status"]
    },
    
    "schedule": {
        "description": """
        定时任务调度工具组（系统按 frequency 重复调度 next_run，与单次日历 add_calendar_event 不同）。
        
        核心功能：
        - 创建定时执行任务（可 once / hourly / daily / weekly）
        - 查看定时任务列表
        - 取消定时任务
        
        适用场景：
        - 每周固定维护、每天检查、按点提醒等重复事项
        - 定期自动执行或触发助手处理
        - 任务调度管理
        """,
        "examples": [
            "每天定时执行",
            "每周固定做一件事",
            "设置一个自动任务",
            "定时提醒我",
            "调度任务",
            "自动化执行",
            "取消定时任务"
        ],
        "tools": ["schedule_task", "schedule_list", "schedule_cancel"]
    },

    # [2026-03-13] keep parity with tool_selector.TOOL_GROUPS so semantic-only routing does not miss groups
    "self_introspection": {
        "description": "列出当前可用工具与能力，回答用户关于「你能做什么、有什么工具」的询问。",
        "examples": ["你有什么能力", "你会什么", "列出你的工具", "你能做什么", "what can you do", "我的工具"],
        "tools": ["list_my_tools"]
    },
    "basic_file": {
        "description": "基础文件读写与搜索：读文件、写文件、重命名/移动、复制文件、建目录、列目录、搜索内容。日记、文档、沙箱 workspace/sandbox。",
        "examples": [
            "读一下这个文件",
            "保存内容到文件",
            "把这个文件改名",
            "复制这个文件到",
            "新建一个文件夹",
            "列出目录",
            "搜索文件里的内容",
            "写日记",
        ],
        "tools": [
            "read_file",
            "write_file",
            "rename_file",
            "list_files",
            "search_files",
            "copy_file",
            "create_directory",
        ],
    },
    "code_execution": {
        "description": "执行 Python 代码或脚本：运行、计算、测试、写个小程序。",
        "examples": ["运行这段代码", "帮我算一下", "执行脚本", "写个脚本计算", "测试这段代码"],
        "tools": ["execute_python"]
    },
    "web_search": {
        "description": "网络搜索与实时信息：查最新消息、新闻、概念解释、网上查询。",
        "examples": ["搜一下", "网上查查", "最新新闻", "什么是XXX", "告诉我关于", "look up"],
        "tools": ["tavily_search"]
    },
    "time_util": {
        "description": "获取当前时间、日期、今天、现在。",
        "examples": ["现在几点", "今天几号", "当前时间", "what time is it", "日期"],
        "tools": ["get_current_time"]
    },
    # [2026-03-13] geo + weather
    "geo_weather": {
        "description": "定位与天气：查询当前位置（城市/经纬度）、实时天气（温度/湿度/风速/天气状况）和近12小时逐小时天气预报。",
        "examples": [
            "今天天气怎么样", "北京天气", "上海明天下雨吗", "现在气温多少",
            "我在哪个城市", "查一下当前位置", "weather today", "what's the weather",
            "帮我查一下深圳天气", "天气预报", "会下雨吗", "需要带伞吗",
        ],
        "tools": ["get_location", "get_weather"]
    },
    "goal_basic": {
        "description": "目标与计划基础：我的目标、目标列表、添加目标、目标详情、删除目标、进度。",
        "examples": ["我的目标", "列出目标", "添加一个目标", "删掉这个目标", "目标进度", "goals"],
        "tools": ["goal_list", "goal_add", "goal_detail", "goal_delete"]
    },
    "learning": {
        "description": "学习与知识库：设定学习目标、记录/搜索/更新/删除知识、学习统计。",
        "examples": ["记住这个", "学到的知识", "搜索我的知识", "删掉这条知识", "修正刚才记错的那条", "设定学习目标", "知识库"],
        "tools": ["set_learning_goal", "add_learned_knowledge", "search_my_knowledge",
                  "update_my_knowledge", "delete_my_knowledge", "get_my_learning_goals", "get_knowledge_stats"]
    },
    "task_planning": {
        "description": "任务与待办：创建任务、列出任务、分解任务、今日计划、执行计划、下一步、删除计划中的某一步、待办、todo。",
        "examples": ["创建任务", "我的待办", "今日计划", "分解任务", "下一步做什么", "删掉计划里多余的一步", "list tasks", "plan today"],
        "tools": ["create_execution_plan", "add_task_to_plan", "delete_plan_task", "get_plan_details",
                  "get_next_task", "start_task", "complete_task", "get_my_active_plans",
                  "create_task", "list_tasks", "decompose_task", "plan_today"]
    },
    "code_proposal": {
        "description": "代码自我改进：提出代码修改建议、列出/检查提案、阅读自己的代码、代码结构分析。",
        "examples": ["提出代码修改", "我的代码提案", "改进这段代码", "读一下我的代码", "自我修改"],
        "tools": ["propose_code_change", "list_my_proposals", "check_proposal_result",
                  "read_my_code", "list_my_code_files", "analyze_code_structure"]
    },
    "moltbook": {
        "description": "Moltbook 社交：发帖、评论、点赞、Feed、搜索、社区、关注、用户资料。",
        "examples": ["在 moltbook 发帖", "评论一下", "点赞", "看 feed", "关注用户", "创建社区"],
        "tools": ["moltbook_post", "moltbook_comment", "moltbook_get_comments",
                  "moltbook_upvote_post", "moltbook_upvote_comment",
                  "moltbook_feed", "moltbook_search", "moltbook_get_post",
                  "moltbook_delete_post", "moltbook_list_communities", "moltbook_join_community",
                  "moltbook_leave_community", "moltbook_create_community",
                  "moltbook_get_my_profile", "moltbook_get_profile",
                  "moltbook_follow_user", "moltbook_unfollow_user"]
    },
    "deep_research": {
        "description": "深度研究与调研：多源搜索、生成研究报告、综合分析、深入了解某主题。",
        "examples": ["深度研究一下", "写一份调研报告", "多角度分析", "深入了解", "investigation report"],
        "tools": ["deep_research", "list_research_reports"]
    },
    "memory_recall": {
        "description": "记忆与历史：回忆之前说的、上次、历史、日记、对话历史、知识库检索。",
        "examples": ["之前我说过什么", "回忆一下", "查日记", "历史记录", "recall memory", "上次对话"],
        "tools": ["recall_memory", "get_recent_context"]
    },
    "visualization": {
        "description": "图表与可视化：画图、折线图、柱状图、饼图、散点图、数据展示、统计图。",
        "examples": ["画个柱状图", "做个折线图", "可视化数据", "create chart", "统计图"],
        "tools": ["create_chart"]
    },
    "data_analysis": {
        "description": "数据统计与数学：平均值、标准差、中位数、方程求解、导数、积分、数学计算。",
        "examples": ["分析数据", "求平均值", "解方程", "算导数", "统计", "solve equation"],
        "tools": ["analyze_data", "solve_math"]
    },
    "browser_unified": {
        "description": "浏览器统一能力：打开网页、截图、网页交互、登录。",
        "examples": ["打开网站", "访问网页", "截图", "网页交互", "点击", "填写"],
        "tools": ["browse_web", "screenshot_page", "browser_interact", "browser_login"]
    },
    "pdf_reader": {
        "description": "PDF：可先用 fetch_url_to_workspace 拉取 https（如 arXiv）到工作区，再 read_pdf / get_pdf_info；支持本地路径搜索与列表。",
        "examples": ["读一下这个 pdf", "arxiv pdf 链接", "下载论文再读", "pdf 里搜索", "论文"],
        "tools": ["fetch_url_to_workspace", "read_pdf", "get_pdf_info", "search_pdf", "list_pdfs"]
    },
    "s_repo_evolution": {
        "description": """
        维护与演进 S 项目本仓库（非仅 sandbox）：读写 backend/config/scripts/docs 等、目录树、仓库内搜索、
        Git 状态/diff/log/add/commit/单文件回滚/打标签，以及在仓库根跑 pytest、python -m compileall（整树语法检查）或 npm test|run|ci。
        个人日记与下载物仍在 workspace/sandbox，用 read_file/write_file；改系统代码用 evolution_*。
        禁止直接改 .env；勿手写 .git/；git push 需人类在本机执行。
        """,
        "examples": [
            "改一下 backend 里某个路由",
            "给 tool_router 加一段分支",
            "跑一遍 pytest",
            "用 compileall 检查 backend 语法",
            "npm test 看一下",
            "git status 有什么改动",
            "看一下 diff",
            "提交一版说明为修复心跳",
            "在仓库里搜 evolution_git",
            "列出 backend 目录结构",
            "从 HEAD 恢复某个文件",
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
    "agent_memory": {
        "description": (
            "外存记忆与可复核记录：把结构化结果写入 workspace/sandbox/agent_memory/runs/，"
            "更新 state.json，并生成 STATUS_SNAPSHOT.md；用户不必手跑脚本。"
        ),
        "examples": [
            "记录一下本轮统计参数和结果",
            "把运行结果落盘",
            "刷新状态快照",
            "同步 agent 外存",
        ],
        "tools": ["agent_memory_record", "agent_memory_sync"],
    },
}


# Legacy keyword buckets kept for backward compatibility (secondary to semantic embeddings)
TOOL_GROUPS_KEYWORDS = {
    "stock": ["股票", "投资", "A股", "美股", "港股", "stock", "invest", "财务", "K线"],
    "chemistry": ["化学", "分子", "SMILES", "化合物", "drug", "molecule", "rdkit", "化学式"],
    "browser": ["浏览器", "网页", "截图", "browser", "navigate", "screenshot", "webpage", "打开网站", "访问"],
    "browser_interaction": ["登录", "login", "发帖", "post", "填写", "表单", "form", "点击", "click", 
                            "输入", "提交", "submit", "注册", "register", "账号", "密码"],
    "email": ["邮件", "发送", "email", "mail", "发邮件", "收件箱", "检查邮件", "新邮件", "未读", "已发送", "发件箱"],
    "file_management": ["删除", "移动", "归档", "重复", "整理", "健康", "状态", "熵", "混乱", "乱",
                        "杂乱", "凌乱", "文件管理", "清理", "工作空间",
                        "delete", "move", "archive", "duplicate", "health", "entropy", "organize", "messy", "cleanup"],
    "code_analysis": ["分析代码", "代码质量", "依赖", "compare", "analyze_python", "质量检查", 
                      "代码", "python", "函数", "类", "import", "沙盒"],
    "calendar": ["日历", "日程", "事件", "提醒", "calendar", "event", "schedule", "时间线"],
    # [2026-03-13] extra geo/weather keyword hints
    "geo_weather": [
        "天气", "气温", "温度", "下雨", "降雨", "下雪", "晴天", "阴天", "多云", "风速", "湿度", "预报",
        "weather", "temperature", "rain", "snow", "forecast", "humidity",
        "定位", "位置", "在哪", "我在哪", "城市", "经纬度", "location", "city",
    ],
    "goal_management": ["目标", "计划", "任务", "里程碑", "goal", "plan", "milestone", "进度"],
    "self_awareness": ["自我", "意识", "内省", "神游", "节律", "self", "inspect", "mind_wandering"],
    "companion": ["情感", "安全", "情绪", "emotion", "safety", "陪伴"],
    "research": ["研究", "暂停研究", "恢复研究", "research"],
    "approval": ["审批", "批准", "approval", "pending"],
    "schedule": ["定时", "调度", "scheduled", "任务调度", "定时任务", "定时提醒", "系统提醒", "schedule_task", "schedule_list", "schedule_cancel", "删除定时", "列出定时", "取消定时"],
    "agent_memory": ["状态快照", "运行记录", "落盘", "复核", "外存", "agent_memory", "同步快照"],
}
