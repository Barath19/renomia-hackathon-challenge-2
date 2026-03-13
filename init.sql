-- Local sidecar DB initialization.
-- Add any tables your solution needs here.
-- NOTE: This DB is ephemeral in Cloud Run (lost on scale-to-zero).
-- Use the training Cloud SQL for persistent data.

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
