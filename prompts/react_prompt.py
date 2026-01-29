"""ReAct (Reasoning + Acting) prompt templates for LLM tool orchestration.

This module provides production-ready prompt templates for implementing the ReAct pattern
in multi-agent systems, improving tool calling success rates through structured reasoning.
"""

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""

    name: str
    template: str
    description: str
    model_type: str = "default"


REACT_SYSTEM_PROMPTS: dict[str, PromptTemplate] = {
    "default": PromptTemplate(
        name="default_react_system",
        template="""You are an intelligent agent that uses tools to complete tasks by following the ReAct (Reasoning + Acting) pattern.

CORE PATTERN:
For each step of reasoning, you MUST follow this exact format:

Thought: [Your reasoning about what to do next]
Action: [Tool name to call]
Action Input: [JSON parameters for the tool]

Wait for the Observation result, then continue with:
Thought: [Analyze the observation and plan next step]
... (repeat as needed)

FINAL FORMAT:
When you have completed the task and have all necessary information:
Thought: [Reason about the final answer]
Final Answer: [Your complete response to the user]

RULES:
1. ALWAYS start with a Thought before any Action
2. Provide Action Input as valid JSON (no markdown code blocks)
3. After each Action, wait for the Observation before proceeding
4. Use only the tools listed below
5. If a tool fails, analyze the error and try a different approach
6. Complete all required steps before providing Final Answer
7. Be specific and detailed in your reasoning

TOOL USAGE GUIDELINES:
- Check tool parameters carefully before calling
- Use default values when optional parameters aren't specified
- Chain multiple tools when needed (output of one becomes input to another)
- Make parallel tool calls when operations are independent
- Handle errors gracefully by analyzing the error message

ERROR HANDLING:
If a tool call fails:
1. Analyze why it failed (missing params, wrong format, etc.)
2. Correct the issue and try again
3. If still failing, try an alternative tool or approach
4. Never give up - find another way to complete the task

AVAILABLE TOOLS:
{{tool_descriptions}}""",
        description="Generic ReAct system prompt for any model",
        model_type="default",
    ),
    "gpt4": PromptTemplate(
        name="gpt4_react_system",
        template="""You are GPT-4, an advanced AI assistant that uses tools effectively through structured reasoning.

REACT FRAMEWORK:
Follow the Thought-Action-Observation cycle systematically:

1. Thought: Analyze the current state and what needs to be done
2. Action: Select the appropriate tool
3. Action Input: Provide parameters as valid JSON
4. Wait for Observation
5. Repeat until task is complete
6. Final Answer: Provide comprehensive response

BEST PRACTICES:
- Think step-by-step before acting
- Verify all required parameters are present
- Use parallel calls when tools are independent
- Chain tools logically when dependencies exist
- Extract relevant information from tool outputs
- Synthesize information from multiple sources

TOOL SELECTION STRATEGY:
- Match tool capabilities to task requirements
- Consider tool descriptions and parameter schemas
- Prefer specific tools over general ones
- Use tools in optimal order (search → filter → transform → store)

ERROR RECOVERY:
- Parse error messages carefully
- Identify root cause (validation, network, logic)
- Apply targeted fixes
- Document learnings in subsequent thoughts

PARALLEL EXECUTION:
When multiple independent operations are needed, structure them as:
Thought: I need to search for X and Y, these are independent
Action: arxiv_search
Action Input: {"query": "X"}
Thought: Now I'll search for Y in parallel
Action: arxiv_search
Action Input: {"query": "Y"}

AVAILABLE TOOLS:
{{tool_descriptions}}""",
        description="GPT-4 optimized ReAct prompt with advanced reasoning guidance",
        model_type="gpt4",
    ),
    "claude": PromptTemplate(
        name="claude_react_system",
        template="""You are Claude, an AI assistant designed for thoughtful tool use and reliable task completion.

REACT METHODOLOGY:
Use this structured approach for complex tasks:

Thought: [Break down what you know and what you need to find out]
Action: [Choose the best tool for the current step]
Action Input: [Provide complete, valid JSON parameters]
Observation: [Review the tool output]
Thought: [Interpret results and determine next action]
...continue until complete...
Final Answer: [Synthesize all findings into a clear response]

THINKING DEEPLY:
- Consider multiple approaches before acting
- Anticipate potential issues and avoid them
- Use tools efficiently (batch operations when possible)
- Verify results before proceeding
- Maintain context across multiple steps

TOOL USAGE PRINCIPLES:
1. Read tool descriptions carefully
2. Validate parameters against schemas
3. Handle edge cases gracefully
4. Extract only relevant information
5. Store intermediate results when beneficial

ERROR HANDLING:
When encountering errors:
- Stay calm and analytical
- Identify the specific failure point
- Implement a targeted fix
- Document what you learned
- Proceed with corrected approach

COMPLEX WORKFLOWS:
For multi-step tasks, maintain a mental model of:
- Current state (what you know)
- Goal state (what you need to achieve)
- Gap analysis (what's missing)
- Plan (how to bridge the gap)

AVAILABLE TOOLS:
{{tool_descriptions}}""",
        description="Claude-optimized ReAct prompt emphasizing careful reasoning",
        model_type="claude",
    ),
    "local_llm": PromptTemplate(
        name="local_llm_react_system",
        template="""You are a helpful AI agent that uses tools to complete tasks by following a simple pattern.

HOW TO WORK:
For each step, follow this format:

Thought: What do I need to do now?
Action: Which tool should I use?
Action Input: What parameters does the tool need? (as JSON)

After the tool runs, you'll see the result. Then:
Thought: What did the result tell me? What should I do next?

Keep doing this until you have all the information you need, then:
Final Answer: Your complete response to the user

SIMPLE RULES:
1. Always write a Thought before using a tool
2. Use only the tools listed below
3. Put Action Input in JSON format like: {"param": "value"}
4. Wait for the result after each tool call
5. If something fails, fix it and try again
6. Don't stop until the task is done

COMMON PATTERNS:
- To search: Action: arxiv_search, Action Input: {"query": "your search term"}
- To store: Action: sqlite_store, Action Input: {"table": "name", "data": {...}}
- To analyze: Action: llm_analyze, Action Input: {"text": "content to analyze"}

IF YOU MAKE A MISTAKE:
That's okay! Just write a Thought about what went wrong and try again with corrected parameters.

AVAILABLE TOOLS:
{{tool_descriptions}}""",
        description="Simplified ReAct prompt for local/smaller models",
        model_type="local_llm",
    ),
}


FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {
    "simple": [
        {
            "name": "Simple ArXiv Search",
            "description": "Single tool call with basic parameters",
            "conversation": [
                {"role": "user", "content": "Search for papers about machine learning"}
            ],
            "reasoning_trace": [
                "Thought: The user wants to find papers about machine learning. I should use the arxiv_search tool with 'machine learning' as the query.",
                "Action: arxiv_search",
                'Action Input: {"query": "machine learning"}',
                "Observation: Found 10 papers including 'Deep Learning' by LeCun et al., 'Machine Learning' by Mitchell, etc.",
                "Thought: I successfully found papers about machine learning. I can now provide the user with the results.",
                "Final Answer: I found 10 papers about machine learning:\n1. Deep Learning - LeCun et al.\n2. Machine Learning - Mitchell\n3. [8 more papers listed...]",
            ],
        }
    ],
    "chaining": [
        {
            "name": "Search, Extract, and Store",
            "description": "Multiple tool calls with data flow between them",
            "conversation": [
                {
                    "role": "user",
                    "content": "Find papers about neural networks, extract their abstracts, and save them to the database",
                }
            ],
            "reasoning_trace": [
                "Thought: I need to complete three steps: search for papers, extract abstracts, and store results. I'll start by searching for neural network papers.",
                "Action: arxiv_search",
                'Action Input: {"query": "neural networks", "max_results": 5}',
                "Observation: Found 5 papers with titles, authors, and full abstracts.",
                "Thought: Now I have the search results. I need to extract just the abstracts from each paper using llm_extract.",
                "Action: llm_extract",
                'Action Input: {"text": "[paper data]", "extract_field": "abstract"}',
                "Observation: Extracted abstracts: [list of abstracts]",
                "Thought: I have the abstracts extracted. Now I need to store them in the database using sqlite_store.",
                "Action: sqlite_store",
                'Action Input: {"table": "neural_network_abstracts", "data": {"abstracts": "[extracted data]"}}',
                "Observation: Successfully stored 5 abstracts in neural_network_abstracts table.",
                "Thought: All three steps are complete. I searched for papers, extracted abstracts, and stored them in the database.",
                "Final Answer: I completed your request:\n1. Found 5 papers about neural networks\n2. Extracted their abstracts\n3. Saved all abstracts to the 'neural_network_abstracts' table in the database",
            ],
        }
    ],
    "error_recovery": [
        {
            "name": "Handle Missing Required Parameter",
            "description": "Recover from validation errors by correcting parameters",
            "conversation": [{"role": "user", "content": "Search for papers"}],
            "reasoning_trace": [
                "Thought: The user wants to search for papers but didn't specify what to search for. I'll try with an empty query.",
                "Action: arxiv_search",
                'Action Input: {"query": ""}',
                "Observation: Error: 'query' parameter is required and cannot be empty.",
                "Thought: The tool failed because the query parameter is required. I need to ask the user what they want to search for.",
                "Final Answer: To search for papers, I need to know what topic you're interested in. Could you please specify what you'd like me to search for?",
            ],
        },
        {
            "name": "Handle Invalid JSON Format",
            "description": "Recover from malformed action inputs",
            "conversation": [
                {"role": "user", "content": "Search for papers about AI with 3 results"}
            ],
            "reasoning_trace": [
                "Thought: I need to search for AI papers with a limit of 3 results. I'll use arxiv_search with the query and max_results parameters.",
                "Action: arxiv_search",
                'Action Input: {query: "AI", max_results: 3}',
                "Observation: Error: Invalid JSON format. Use double quotes for keys and string values.",
                "Thought: I made a formatting error - I used single quotes instead of double quotes. Let me fix the JSON format.",
                "Action: arxiv_search",
                'Action Input: {"query": "AI", "max_results": 3}',
                "Observation: Found 3 papers about AI: [paper list]",
                "Thought: The corrected JSON format worked and I successfully found 3 AI papers.",
                "Final Answer: I found 3 papers about AI:\n1. Artificial Intelligence: A Modern Approach\n2. AI Safety Research\n3. Machine Learning for AI Systems",
            ],
        },
    ],
    "parallel": [
        {
            "name": "Independent Parallel Searches",
            "description": "Execute multiple independent tool calls simultaneously",
            "conversation": [
                {
                    "role": "user",
                    "content": "Search for papers about reinforcement learning and computer vision simultaneously",
                }
            ],
            "reasoning_trace": [
                "Thought: The user wants two searches: one for reinforcement learning and one for computer vision. These are independent, so I can make both searches.",
                "Action: arxiv_search",
                'Action Input: {"query": "reinforcement learning"}',
                "Thought: Now I'll search for computer vision papers while the first search is processing.",
                "Action: arxiv_search",
                'Action Input: {"query": "computer vision"}',
                "Observation (from first search): Found 10 papers on reinforcement learning including [list].",
                "Observation (from second search): Found 10 papers on computer vision including [list].",
                "Thought: Both searches completed successfully. I have results for both topics.",
                "Final Answer: I completed both searches:\n\n**Reinforcement Learning Papers:**\n1. Deep Q-Networks\n2. Policy Gradient Methods\n3. Actor-Critic Algorithms\n[7 more papers...]\n\n**Computer Vision Papers:**\n1. Convolutional Neural Networks\n2. Object Detection with YOLO\n3. Image Segmentation with U-Net\n[7 more papers...]",
            ],
        },
        {
            "name": "Parallel Data Collection",
            "description": "Collect data from multiple sources in parallel",
            "conversation": [
                {
                    "role": "user",
                    "content": "Get recent papers on transformers and attention mechanisms from both AI and ML categories",
                }
            ],
            "reasoning_trace": [
                "Thought: I need to search for transformers and attention papers in two different categories. These searches are independent of each other.",
                "Action: arxiv_search",
                'Action Input: {"query": "transformers attention", "categories": ["cs.AI"], "max_results": 5}',
                "Thought: While that search runs, I'll also search in the ML category.",
                "Action: arxiv_search",
                'Action Input: {"query": "transformers attention", "categories": ["cs.LG"], "max_results": 5}',
                "Observation (cs.AI): Found 5 papers: [AI category papers]",
                "Observation (cs.LG): Found 5 papers: [ML category papers]",
                "Thought: I have collected papers from both categories. Now I can present the combined results.",
                "Final Answer: I found papers from both categories:\n\n**AI Category (cs.AI):**\n- Attention Is All You Need\n- BERT: Pre-training of Deep Bidirectional Transformers\n- [3 more papers...]\n\n**Machine Learning Category (cs.LG):**\n- Transformer-XL\n- Longformer: The Long-Document Transformer\n- [3 more papers...]\n\nTotal: 10 relevant papers found.",
            ],
        },
    ],
    "complex": [
        {
            "name": "Multi-Step Research Workflow",
            "description": "Complex reasoning with multiple chained operations",
            "conversation": [
                {
                    "role": "user",
                    "content": "Research recent papers on transformer architectures, save them to the database, and then summarize the key findings",
                }
            ],
            "reasoning_trace": [
                "Thought: This is a complex task requiring three phases: 1) Search for transformer papers, 2) Store results in database, 3) Summarize findings. I'll start with the search.",
                "Action: arxiv_search",
                'Action Input: {"query": "transformer architectures", "categories": ["cs.AI", "cs.LG", "cs.CL"], "max_results": 10, "date_from": "2023-01-01"}',
                "Observation: Found 10 recent papers including 'Efficient Transformers', 'Vision Transformers', etc. with full metadata.",
                "Thought: Good, I have the search results. Now I need to store this data in the database. I'll create a structured entry with all relevant information.",
                "Action: sqlite_store",
                'Action Input: {"table": "transformer_papers", "data": {"papers": "[paper data]", "search_date": "2024-01-24", "query": "transformer architectures"}}',
                "Observation: Successfully stored 10 papers in transformer_papers table.",
                "Thought: Papers are now stored. Next, I need to summarize the key findings from these papers. I'll extract the main content and pass it to the summarizer.",
                "Action: llm_summarize",
                'Action Input: {"text": "[combined abstracts and content from all papers]", "max_length": 500}',
                "Observation: Summary generated: Recent transformer research focuses on efficiency improvements, vision applications, and long-context handling...",
                "Thought: All three phases are complete. I have searched, stored, and summarized the transformer architecture papers.",
                "Final Answer: I completed your research task:\n\n**1. Search Results:**\nFound 10 recent papers on transformer architectures from arXiv (2023-2024), covering topics like efficient transformers, vision transformers, and long-context models.\n\n**2. Database Storage:**\nAll 10 papers have been saved to the 'transformer_papers' table with full metadata, including titles, authors, abstracts, and publication dates.\n\n**3. Key Findings Summary:**\nRecent transformer research focuses on three main areas:\n- **Efficiency:** New architectures reducing computational complexity while maintaining performance\n- **Vision Applications:** Adapting transformers for image and video processing tasks\n- **Long-Context Handling:** Techniques to process longer sequences without memory issues\n\nThe papers suggest transformers continue to evolve beyond NLP into broader AI applications.",
            ],
        },
        {
            "name": "Context-Dependent Analysis",
            "description": "Use conversation context to inform tool selection",
            "conversation": [
                {
                    "role": "user",
                    "content": "I found some papers, can you help me analyze them?",
                },
                {
                    "role": "assistant",
                    "content": "I can help you analyze papers using the LLM. Please provide the paper content or titles.",
                },
                {
                    "role": "user",
                    "content": "Here are the papers: [Paper A: Attention is all you need, Paper B: BERT: Pre-training of Deep Bidirectional Transformers]",
                },
            ],
            "reasoning_trace": [
                "Thought: The user has provided two specific paper titles and wants analysis. Since they've already identified the papers, I should use the analysis tool rather than searching. I'll analyze both papers together.",
                "Action: llm_analyze",
                'Action Input: {"text": "Paper A: Attention is all you need. Paper B: BERT: Pre-training of Deep Bidirectional Transformers", "analysis_type": "comparative"}',
                "Observation: Analysis complete. Both papers are foundational transformer works. Paper A introduced the transformer architecture, while Paper B applied it to pre-training for NLP tasks.",
                "Thought: The analysis provides good context. I should now give the user a comprehensive answer about these papers.",
                "Final Answer: I've analyzed both papers:\n\n**Paper A: 'Attention Is All You Need' (Vaswani et al., 2017)**\n- Introduced the transformer architecture\n- Key innovation: Self-attention mechanism\n- Replaced RNNs/LSTMs in sequence modeling\n- Foundation for modern NLP\n\n**Paper B: 'BERT' (Devlin et al., 2018)**\n- Built on transformer architecture\n- Introduced bidirectional pre-training\n- Achieved SOTA on 11 NLP tasks\n- Popularized fine-tuning approach\n\n**Connection:** BERT is a direct application of the transformer architecture from Paper A, demonstrating how the innovation enabled breakthrough practical applications.",
            ],
        },
        {
            "name": "Database Query and Transformation",
            "description": "Complex workflow with database operations",
            "conversation": [
                {
                    "role": "user",
                    "content": "Query the papers table for entries from 2024 and store the results in a new table called recent_papers",
                }
            ],
            "reasoning_trace": [
                "Thought: I need to perform two database operations: 1) Query the papers table for 2024 entries, 2) Create a new table with those results. I'll start with the query.",
                "Action: sqlite_query",
                'Action Input: {"table": "papers", "filters": {"year": "2024"}}',
                "Observation: Query returned 25 papers from 2024 with metadata including titles, authors, and publication dates.",
                "Thought: I have the 2024 papers. Now I need to store these in a new table called 'recent_papers'. I'll structure the data appropriately.",
                "Action: sqlite_store",
                'Action Input: {"table": "recent_papers", "data": {"papers": "[25 paper records]", "query_date": "2024-01-24", "source_table": "papers"}}',
                "Observation: Successfully created recent_papers table and stored 25 papers.",
                "Thought: Both operations completed successfully. I queried the original table and created a new filtered table.",
                "Final Answer: Task completed successfully:\n\n1. Queried the 'papers' table for 2024 entries\n2. Found 25 papers from 2024\n3. Created new table 'recent_papers'\n4. Stored all 25 papers in the new table with metadata\n\nThe recent_papers table now contains all 2024 papers from the original papers table, ready for further analysis.",
            ],
        },
    ],
}


def get_react_system_prompt(
    model_type: str = "default", tool_descriptions: list[str] | None = None
) -> str:
    """Get a ReAct system prompt for a specific model type.

    Args:
        model_type: The type of model (default, gpt4, claude, local_llm)
        tool_descriptions: Optional list of tool descriptions to include

    Returns:
        Formatted system prompt with tool descriptions
    """
    if model_type not in REACT_SYSTEM_PROMPTS:
        model_type = "default"

    template = REACT_SYSTEM_PROMPTS[model_type].template

    if tool_descriptions:
        tool_desc_text = "\n".join(f"- {desc}" for desc in tool_descriptions)
    else:
        tool_desc_text = "No tools available"

    return template.replace("{{tool_descriptions}}", tool_desc_text)


def get_few_shot_examples(category: str | None = None) -> list[dict]:
    """Get few-shot examples for ReAct prompting.

    Args:
        category: Category of examples to return (simple, chaining, error_recovery, parallel, complex)
                 If None, returns all examples

    Returns:
        List of example dictionaries
    """
    if category:
        return FEW_SHOT_EXAMPLES.get(category, [])

    all_examples = []
    for examples in FEW_SHOT_EXAMPLES.values():
        all_examples.extend(examples)
    return all_examples


def format_example(example: dict) -> str:
    """Format a single example as a demonstration string.

    Args:
        example: Example dictionary with conversation and reasoning_trace

    Returns:
        Formatted example string
    """
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"Example: {example.get('name', 'Unnamed')}")
    lines.append(f"Description: {example.get('description', 'No description')}")
    lines.append(f"{'=' * 60}\n")

    for msg in example.get("conversation", []):
        lines.append(f"{msg['role'].upper()}: {msg['content']}")

    lines.append("\nReasoning Trace:")
    for step in example.get("reasoning_trace", []):
        lines.append(f"  {step}")

    return "\n".join(lines)


def build_react_prompt(
    messages: list[dict],
    tools: list[dict],
    model_type: str = "default",
    include_examples: bool = True,
    example_categories: list[str] | None = None,
) -> list[dict]:
    """Build a complete ReAct prompt with system message, examples, and conversation.

    Args:
        messages: The conversation messages (user/assistant history)
        tools: List of tool schemas in OpenAI function format
        model_type: Type of model for prompt optimization
        include_examples: Whether to include few-shot examples
        example_categories: Categories of examples to include (default: all)

    Returns:
        Complete message list for LLM API call
    """
    tool_descriptions = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        tool_descriptions.append(f"{name}: {desc}")

    system_prompt = get_react_system_prompt(model_type, tool_descriptions)

    result_messages = [{"role": "system", "content": system_prompt}]

    if include_examples:
        if example_categories:
            examples = []
            for cat in example_categories:
                examples.extend(get_few_shot_examples(cat))
        else:
            examples = get_few_shot_examples()

        if examples:
            example_text = "\n\n".join(format_example(ex) for ex in examples[:3])
            result_messages.append(
                {
                    "role": "system",
                    "content": f"EXAMPLES OF HOW TO USE TOOLS:\n{example_text}",
                }
            )

    result_messages.extend(messages)

    return result_messages


def get_example_by_name(name: str) -> dict | None:
    """Find a specific example by name.

    Args:
        name: Name of the example to find

    Returns:
        Example dictionary or None if not found
    """
    for examples in FEW_SHOT_EXAMPLES.values():
        for example in examples:
            if example.get("name") == name:
                return example
    return None


def validate_prompt_structure(messages: list[dict]) -> bool:
    """Validate that the prompt has proper ReAct structure.

    Args:
        messages: Message list to validate

    Returns:
        True if structure is valid, False otherwise
    """
    if not messages:
        return False

    has_system = any(msg.get("role") == "system" for msg in messages)
    has_user = any(msg.get("role") == "user" for msg in messages)

    return has_system and has_user


def extract_tool_descriptions(tools: list[dict]) -> list[str]:
    """Extract human-readable descriptions from tool schemas.

    Args:
        tools: List of tool schemas

    Returns:
        List of description strings
    """
    descriptions = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        param_info = []
        if required:
            param_info.append(f"Required: {', '.join(required)}")
        if params:
            optional = [p for p in params.keys() if p not in required]
            if optional:
                param_info.append(f"Optional: {', '.join(optional)}")

        if param_info:
            desc += f" ({'; '.join(param_info)})"

        descriptions.append(f"{name}: {desc}")

    return descriptions


def build_tool_focused_prompt(
    tools: list[dict], task_description: str, model_type: str = "default"
) -> str:
    """Build a prompt focused on a specific task with relevant tools.

    Args:
        tools: List of available tools
        task_description: Description of the task to complete
        model_type: Type of model for prompt optimization

    Returns:
        Complete prompt string
    """
    tool_descriptions = extract_tool_descriptions(tools)
    system_prompt = get_react_system_prompt(model_type, tool_descriptions)

    return f"""{system_prompt}

Your task: {task_description}

Begin by thinking about what you need to do, then proceed with the ReAct pattern."""


if __name__ == "__main__":
    print("ReAct Prompt Templates Module")
    print("=" * 50)
    print(f"Available system prompts: {list(REACT_SYSTEM_PROMPTS.keys())}")
    print(f"Available example categories: {list(FEW_SHOT_EXAMPLES.keys())}")
    print(f"Total examples: {sum(len(ex) for ex in FEW_SHOT_EXAMPLES.values())}")
