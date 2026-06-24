from smolagents import ToolCallingAgent, TransformersModel, tool


COURSE_NOTES = {
    "agent": """
AI Agent 是由大模型驱动、能够根据目标选择工具并执行多步任务的程序。
它通常由模型、系统提示词、工具、状态和评估机制组成。
""",
    "rag": """
RAG 是检索增强生成。
Agent 先从外部知识库检索相关资料，再基于资料生成答案，
从而减少模型仅凭参数记忆回答时的幻觉。
""",
    "tool": """
工具是 Agent 可以调用的外部能力，例如读取文件、搜索网页、
查询数据库、执行 Python、调用 API。
模型决定是否调用工具，程序实际执行工具。
""",
}


@tool
def get_course_note(topic: str) -> str:
    """
    查询 AI Agent 课程笔记。

    Args:
        topic: 课程主题，可选 agent、rag、tool。
    """
    key = topic.lower().strip()
    return COURSE_NOTES.get(
        key,
        "没有这个主题。可选：agent、rag、tool。",
    )


@tool
def list_topics() -> str:
    """
    列出当前课程可查询的主题。

    Args:
        无参数。
    """
    return "当前可学习主题：agent、rag、tool。"


def main() -> None:
    model = TransformersModel(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        device_map="auto",
    )

    agent = ToolCallingAgent(
        tools=[get_course_note, list_topics],
        model=model,
        instructions="""
你是一名严谨的 AI Agent 课程助教。

规则：
1. 用户询问课程知识时，优先使用工具查询资料。
2. 不要伪造工具中不存在的课程内容。
3. 回答按“概念、例子、练习”组织。
4. 回答使用中文。
""",
        max_steps=4,
    )

    result = agent.run("解释 RAG，并说明它和普通聊天机器人的区别。")
    print(result)


if __name__ == "__main__":
    main()