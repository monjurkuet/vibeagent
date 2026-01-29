TEST_CASES = [
    {
        "name": "Simple tool call - ArXiv search",
        "messages": [{"role": "user", "content": "Search for papers about machine learning"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        "expected_tool": "arxiv_search",
        "expected_params": {"query": "machine learning"},
    },
    {
        "name": "Multiple tool calls - Search and save papers",
        "messages": [
            {
                "role": "user",
                "content": "Search for papers about neural networks and save the results to the database",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sqlite_store",
                    "description": "Store data in SQLite database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Table name"},
                            "data": {"type": "object", "description": "Data to store"},
                        },
                        "required": ["table", "data"],
                    },
                },
            },
        ],
        "expected_tool": ["arxiv_search", "sqlite_store"],
    },
    {
        "name": "No tools needed - General question",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        "expect_no_tools": True,
    },
    {
        "name": "Complex multi-step - Research and summarize",
        "messages": [
            {
                "role": "user",
                "content": "Research recent papers on transformer architectures, save them to the database, and then summarize the key findings",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sqlite_store",
                    "description": "Store data in SQLite database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Table name"},
                            "data": {"type": "object", "description": "Data to store"},
                        },
                        "required": ["table", "data"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "llm_summarize",
                    "description": "Summarize text using LLM",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to summarize",
                            },
                            "max_length": {"type": "integer", "default": 200},
                        },
                        "required": ["text"],
                    },
                },
            },
        ],
        "expected_tool": ["arxiv_search", "sqlite_store", "llm_summarize"],
    },
    {
        "name": "Parallel tool calls - Multiple searches at once",
        "messages": [
            {
                "role": "user",
                "content": "Search for papers about reinforcement learning and computer vision simultaneously",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        "expected_tool": ["arxiv_search", "arxiv_search"],
        "expected_params": [
            {"query": "reinforcement learning"},
            {"query": "computer vision"},
        ],
    },
    {
        "name": "Error handling - Invalid parameters",
        "messages": [{"role": "user", "content": "Search for papers"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        "expect_no_tools": True,
    },
    {
        "name": "Tool call with optional parameters",
        "messages": [
            {
                "role": "user",
                "content": "Search for 5 papers about deep learning in AI category",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        "expected_tool": "arxiv_search",
        "expected_params": {
            "query": "deep learning",
            "max_results": 5,
            "categories": ["cs.AI"],
        },
    },
    {
        "name": "Database query and storage",
        "messages": [
            {
                "role": "user",
                "content": "Query the papers table for entries from 2024 and store the results in a new table called recent_papers",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "sqlite_query",
                    "description": "Query SQLite database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Table name"},
                            "filters": {
                                "type": "object",
                                "description": "Query filters",
                            },
                        },
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sqlite_store",
                    "description": "Store data in SQLite database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Table name"},
                            "data": {"type": "object", "description": "Data to store"},
                        },
                        "required": ["table", "data"],
                    },
                },
            },
        ],
        "expected_tool": ["sqlite_query", "sqlite_store"],
    },
    {
        "name": "Context-dependent tool selection",
        "messages": [
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
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "llm_analyze",
                    "description": "Analyze text using LLM",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze",
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
        ],
        "expected_tool": "llm_analyze",
    },
    {
        "name": "Tool chaining - Search, process, store",
        "messages": [
            {
                "role": "user",
                "content": "Find papers about generative AI, extract their abstracts, and save them to the database",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "Search for papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "llm_extract",
                    "description": "Extract information using LLM",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to process",
                            },
                            "extract_field": {
                                "type": "string",
                                "description": "Field to extract",
                            },
                        },
                        "required": ["text", "extract_field"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sqlite_store",
                    "description": "Store data in SQLite database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Table name"},
                            "data": {"type": "object", "description": "Data to store"},
                        },
                        "required": ["table", "data"],
                    },
                },
            },
        ],
        "expected_tool": ["arxiv_search", "llm_extract", "sqlite_store"],
    },
]
