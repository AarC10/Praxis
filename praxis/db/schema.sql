-- Core skills
CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    short_desc TEXT NOT NULL,
    long_desc TEXT,
    category TEXT,
    created_at TIMESTAMP NOT NULL,
    created_by TEXT NOT NULL,
    active_version_id TEXT
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_created_by ON skills(created_by);

-- Skill versions
CREATE TABLE IF NOT EXISTS skill_versions (
    version_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('draft', 'active', 'failed')),
    created_at TIMESTAMP NOT NULL,
    code_path TEXT NOT NULL,
    contract_name TEXT NOT NULL,
    embedding BLOB,
    FOREIGN KEY(skill_id) REFERENCES skills(skill_id),
    UNIQUE(skill_id, version)
);

CREATE INDEX idx_versions_skill ON skill_versions(skill_id);
CREATE INDEX idx_versions_status ON skill_versions(status);

-- Skill interfacing contracts
CREATE TABLE IF NOT EXISTS skill_interfaces (
    interface_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    inputs_schema TEXT NOT NULL,
    outputs_schema TEXT NOT NULL,
    termination_condition TEXT,
    failure_modes TEXT,
    FOREIGN KEY(version_id) REFERENCES skill_versions(version_id)
);

CREATE INDEX idx_interfaces_version ON skill_interfaces(version_id);

-- Execution log
CREATE TABLE IF NOT EXISTS skill_executions (
    exec_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    exit_status TEXT CHECK(exit_status IN ('running', 'success', 'failure', 'timeout')),
    failure_reason TEXT,

    -- Performance metrics
    latency_ms INTEGER,
    cpu_ms INTEGER,
    mem_peak_mb INTEGER,

    -- Generation context
    llm_model TEXT,
    llm_tokens_input INTEGER,
    llm_tokens_output INTEGER,
    llm_latency_ms INTEGER,
    generation_attempts INTEGER,

    -- Request context
    user_request TEXT,

    FOREIGN KEY(version_id) REFERENCES skill_versions(version_id)
);

CREATE INDEX idx_executions_version ON skill_executions(version_id);
CREATE INDEX idx_executions_status ON skill_executions(exit_status);
CREATE INDEX idx_executions_time ON skill_executions(start_time);
CREATE INDEX idx_executions_llm ON skill_executions(llm_model);

-- Skill artifacts
CREATE TABLE IF NOT EXISTS skill_artifacts (
    artifact_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK(artifact_type IN ('source', 'binary', 'model', 'config')),
    path TEXT NOT NULL,
    checksum TEXT,
    FOREIGN KEY(version_id) REFERENCES skill_versions(version_id)
);

CREATE INDEX idx_artifacts_version ON skill_artifacts(version_id);

-- Metrics for analysis
CREATE TABLE IF NOT EXISTS skill_metrics (
    metric_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    exec_id TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata TEXT,
    FOREIGN KEY(version_id) REFERENCES skill_versions(version_id),
    FOREIGN KEY(exec_id) REFERENCES skill_executions(exec_id)
);

CREATE INDEX idx_metrics_version ON skill_metrics(version_id, metric_name);
CREATE INDEX idx_metrics_exec ON skill_metrics(exec_id);
CREATE INDEX idx_metrics_name ON skill_metrics(metric_name);

-- Skill dependencies (for composition)
CREATE TABLE IF NOT EXISTS skill_dependencies (
    dependency_id TEXT PRIMARY KEY,
    parent_version TEXT NOT NULL,
    child_version TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('uses', 'wraps', 'extends')),
    FOREIGN KEY(parent_version) REFERENCES skill_versions(version_id),
    FOREIGN KEY(child_version) REFERENCES skill_versions(version_id),
    UNIQUE(parent_version, child_version)
);

CREATE INDEX idx_dependencies_parent ON skill_dependencies(parent_version);
CREATE INDEX idx_dependencies_child ON skill_dependencies(child_version);