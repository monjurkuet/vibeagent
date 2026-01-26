-- VibeAgent Database Schema
-- SQLite database schema for tracking sessions, messages, tool calls, and test results

-- Sessions table: tracks agent interaction sessions
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    session_type TEXT NOT NULL,
    model TEXT NOT NULL,
    orchestrator_type TEXT,
    total_iterations INTEGER DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,
    total_duration_ms INTEGER DEFAULT 0,
    final_status TEXT,
    error_message TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table: stores all messages in a session
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    raw_content TEXT,
    message_index INTEGER NOT NULL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    model TEXT,
    temperature REAL,
    max_tokens INTEGER,
    parent_message_id INTEGER,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_message_id) REFERENCES messages(id)
);

-- LLM responses table: tracks LLM response metrics
CREATE TABLE IF NOT EXISTS llm_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    message_id INTEGER,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    finish_reason TEXT,
    raw_response TEXT,
    reasoning_content TEXT,
    tool_calls_count INTEGER DEFAULT 0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- Tool calls table: tracks all tool executions
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    call_index INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    parameters TEXT NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    error_type TEXT,
    retry_count INTEGER DEFAULT 0,
    is_parallel BOOLEAN DEFAULT 0,
    parallel_batch_id INTEGER,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Tool results table: stores tool execution results
CREATE TABLE IF NOT EXISTS tool_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_call_id INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    data TEXT,
    error TEXT,
    result_size_bytes INTEGER,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id) ON DELETE CASCADE
);

-- Test cases table: stores test case definitions
CREATE TABLE IF NOT EXISTS test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    messages TEXT NOT NULL,
    tools TEXT NOT NULL,
    expected_tools TEXT,
    expected_parameters TEXT,
    expect_no_tools BOOLEAN DEFAULT 0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Test runs table: tracks test execution runs
CREATE TABLE IF NOT EXISTS test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_case_id INTEGER NOT NULL,
    session_id INTEGER,
    run_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    total_iterations INTEGER,
    total_tool_calls INTEGER,
    final_status TEXT,
    error_message TEXT,
    metadata TEXT,
    FOREIGN KEY (test_case_id) REFERENCES test_cases(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Judge evaluations table: stores LLM judge evaluations
CREATE TABLE IF NOT EXISTS judge_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id INTEGER NOT NULL,
    tool_call_id INTEGER,
    judge_model TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT NOT NULL,
    evaluation_type TEXT DEFAULT 'semantic',
    criteria TEXT,
    details TEXT,
    evaluation_time_ms INTEGER,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- Reasoning steps table: tracks reasoning steps (ReAct/ToT)
CREATE TABLE IF NOT EXISTS reasoning_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_call_id INTEGER,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- Error recovery table: tracks error recovery attempts
CREATE TABLE IF NOT EXISTS error_recovery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    tool_call_id INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    recovery_strategy TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    original_error TEXT NOT NULL,
    recovery_details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- Self-correction table: tracks self-correction attempts
CREATE TABLE IF NOT EXISTS self_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    attempt_number INTEGER NOT NULL,
    trigger TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    strategy_description TEXT NOT NULL,
    original_error TEXT,
    success BOOLEAN NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    reflection_summary TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Performance metrics table: stores performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_index ON messages(session_id, message_index);
CREATE INDEX IF NOT EXISTS idx_llm_responses_session_id ON llm_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_test_runs_test_case_id ON test_runs(test_case_id);
CREATE INDEX IF NOT EXISTS idx_test_runs_status ON test_runs(status);
CREATE INDEX IF NOT EXISTS idx_judge_evaluations_test_run_id ON judge_evaluations(test_run_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_session_id ON performance_metrics(session_id);

-- Views for common queries
CREATE VIEW IF NOT EXISTS test_performance AS
SELECT 
    tc.id as test_case_id,
    tc.name as test_case_name,
    tc.category,
    COUNT(tr.id) as total_runs,
    COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) as successful_runs,
    ROUND(CAST(COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) AS FLOAT) / NULLIF(COUNT(tr.id), 0) * 100, 2) as success_rate,
    MAX(tr.completed_at) as last_run
FROM test_cases tc
LEFT JOIN test_runs tr ON tc.id = tr.test_case_id
GROUP BY tc.id;

CREATE VIEW IF NOT EXISTS tool_success_rate AS
SELECT 
    tool_name,
    COUNT(*) as total_calls,
    COUNT(CASE WHEN success = 1 THEN 1 END) as successful_calls,
    ROUND(CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT) / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
    AVG(execution_time_ms) as avg_execution_time_ms
FROM tool_calls
GROUP BY tool_name;

CREATE VIEW IF NOT EXISTS model_comparison AS
SELECT 
    model,
    COUNT(*) as total_requests,
    SUM(prompt_tokens) as total_prompt_tokens,
    SUM(completion_tokens) as total_completion_tokens,
    SUM(total_tokens) as total_tokens,
    AVG(response_time_ms) as avg_response_time_ms,
    ROUND(CAST(SUM(completion_tokens) AS FLOAT) / NULLIF(SUM(prompt_tokens), 0), 2) as avg_token_efficiency
FROM llm_responses
GROUP BY model;
