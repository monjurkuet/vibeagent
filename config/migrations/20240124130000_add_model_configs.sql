-- Migration for model configuration storage
-- Adds tables for storing model configs, performance tracking, and A/B testing

-- Model configurations table: stores model-specific configurations
CREATE TABLE IF NOT EXISTS model_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT UNIQUE NOT NULL,
    model_family TEXT NOT NULL,
    display_name TEXT NOT NULL,
    config_data TEXT NOT NULL,
    config_version TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Configuration versions table: tracks version history
CREATE TABLE IF NOT EXISTS model_config_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    config_version TEXT NOT NULL,
    config_data TEXT NOT NULL,
    version_hash TEXT NOT NULL,
    change_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, config_version)
);

-- Model performance metrics table: tracks performance by configuration
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    config_version TEXT NOT NULL,
    phase TEXT NOT NULL,
    temperature REAL NOT NULL,
    max_tokens INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    duration_ms REAL NOT NULL,
    iterations INTEGER NOT NULL,
    tokens_used INTEGER,
    session_id INTEGER,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- A/B test configurations table: stores A/B test setups
CREATE TABLE IF NOT EXISTS ab_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    base_config_version TEXT NOT NULL,
    variant_configs TEXT NOT NULL,
    traffic_split TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- A/B test results table: tracks A/B test performance
CREATE TABLE IF NOT EXISTS ab_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER NOT NULL,
    variant_name TEXT NOT NULL,
    session_id INTEGER,
    success BOOLEAN NOT NULL,
    duration_ms REAL NOT NULL,
    user_rating INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (test_id) REFERENCES ab_tests(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Configuration optimization suggestions table: stores generated suggestions
CREATE TABLE IF NOT EXISTS config_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    phase TEXT NOT NULL,
    suggestion_type TEXT NOT NULL,
    current_value REAL,
    suggested_value REAL,
    expected_improvement REAL,
    confidence REAL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_configs_name ON model_configs(model_name);
CREATE INDEX IF NOT EXISTS idx_model_configs_family ON model_configs(model_family);
CREATE INDEX IF NOT EXISTS idx_model_config_versions_name ON model_config_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_phase ON model_performance(phase);
CREATE INDEX IF NOT EXISTS idx_ab_tests_model ON ab_tests(model_name);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_test ON ab_test_results(test_id);
CREATE INDEX IF NOT EXISTS idx_config_suggestions_model ON config_suggestions(model_name);

-- Views for common queries
CREATE VIEW IF NOT EXISTS model_config_summary AS
SELECT
    mc.model_name,
    mc.model_family,
    mc.display_name,
    mc.config_version,
    mc.is_active,
    COUNT(DISTINCT mcv.config_version) as total_versions,
    MAX(mcv.created_at) as last_updated,
    AVG(CASE WHEN mp.success = 1 THEN 1.0 ELSE 0.0 END) as avg_success_rate,
    AVG(mp.duration_ms) as avg_duration_ms
FROM model_configs mc
LEFT JOIN model_config_versions mcv ON mc.model_name = mcv.model_name
LEFT JOIN model_performance mp ON mc.model_name = mp.model_name AND mc.config_version = mp.config_version
GROUP BY mc.model_name;

CREATE VIEW IF NOT EXISTS ab_test_performance AS
SELECT
    at.test_name,
    at.model_name,
    at.status,
    COUNT(DISTINCT atr.session_id) as total_sessions,
    atr.variant_name,
    AVG(CASE WHEN atr.success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
    AVG(atr.duration_ms) as avg_duration_ms,
    AVG(atr.user_rating) as avg_user_rating
FROM ab_tests at
LEFT JOIN ab_test_results atr ON at.id = atr.test_id
GROUP BY at.test_name, atr.variant_name;

CREATE VIEW IF NOT EXISTS phase_performance_comparison AS
SELECT
    model_name,
    phase,
    ROUND(temperature, 1) as temp_bucket,
    COUNT(*) as total_runs,
    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
    AVG(duration_ms) as avg_duration,
    AVG(iterations) as avg_iterations
FROM model_performance
GROUP BY model_name, phase, ROUND(temperature, 1)
ORDER BY model_name, phase, success_rate DESC;
