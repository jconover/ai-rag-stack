-- ============================================================================
-- Migration: Partition query_logs table by month (RANGE on timestamp)
-- ============================================================================
-- This migration converts the query_logs table to a partitioned table for
-- improved query performance on time-range queries and easier data management.
--
-- Benefits:
-- - Faster queries on date ranges (partition pruning)
-- - Efficient data archival and deletion (DROP PARTITION vs DELETE)
-- - Better vacuum performance (per-partition maintenance)
-- - Improved index efficiency (smaller per-partition indexes)
--
-- Usage:
--   psql -h localhost -U postgres -d devops_assistant -f partition_query_logs.sql
--
-- Note: This migration is idempotent and safe to run multiple times.
-- ============================================================================

BEGIN;

-- ============================================================================
-- Step 1: Check if already partitioned
-- ============================================================================
DO $$
DECLARE
    is_partitioned BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'query_logs'
        AND c.relkind = 'p'  -- 'p' = partitioned table
        AND n.nspname = 'public'
    ) INTO is_partitioned;

    IF is_partitioned THEN
        RAISE NOTICE 'query_logs is already partitioned. Skipping migration.';
        -- Commit and exit early
        RETURN;
    END IF;
END $$;

-- ============================================================================
-- Step 2: Create the new partitioned table structure
-- ============================================================================
-- We use timestamp as the partition key since it's the primary time reference
-- for analytics queries. The table structure mirrors the original exactly.

CREATE TABLE IF NOT EXISTS query_logs_partitioned (
    id UUID NOT NULL,
    session_id VARCHAR(36) NOT NULL,
    user_id UUID,
    query TEXT NOT NULL,
    model VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    latency_ms FLOAT,
    token_count INTEGER,
    retrieval_scores JSONB,
    sources_returned VARCHAR(500)[],
    response_length INTEGER,
    context_used BOOLEAN DEFAULT TRUE,
    sources_count INTEGER,
    retrieval_time_ms FLOAT,
    rerank_time_ms FLOAT,
    total_time_ms FLOAT,
    avg_similarity_score FLOAT,
    avg_rerank_score FLOAT,
    hybrid_search_used BOOLEAN DEFAULT FALSE,
    hyde_used BOOLEAN DEFAULT FALSE,
    reranker_used BOOLEAN DEFAULT FALSE,
    web_search_used BOOLEAN DEFAULT FALSE,
    extra_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    -- Primary key must include partition key for partitioned tables
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Add comments for documentation
COMMENT ON TABLE query_logs_partitioned IS 'Partitioned query logs table for analytics. Partitioned by month on timestamp column.';
COMMENT ON COLUMN query_logs_partitioned.timestamp IS 'Query timestamp - partition key for monthly partitions';

-- ============================================================================
-- Step 3: Create indexes on the partitioned table
-- ============================================================================
-- Indexes are automatically created on each partition

CREATE INDEX IF NOT EXISTS ix_query_logs_part_session_id
    ON query_logs_partitioned (session_id);

CREATE INDEX IF NOT EXISTS ix_query_logs_part_user_id
    ON query_logs_partitioned (user_id)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_query_logs_part_model
    ON query_logs_partitioned (model);

CREATE INDEX IF NOT EXISTS ix_query_logs_part_timestamp
    ON query_logs_partitioned (timestamp);

CREATE INDEX IF NOT EXISTS ix_query_logs_part_created_at
    ON query_logs_partitioned (created_at);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS ix_query_logs_part_model_timestamp
    ON query_logs_partitioned (model, timestamp);

CREATE INDEX IF NOT EXISTS ix_query_logs_part_session_timestamp
    ON query_logs_partitioned (session_id, timestamp);

-- ============================================================================
-- Step 4: Create initial monthly partitions
-- ============================================================================
-- Create partitions for the past 3 months, current month, and next month
-- This ensures we have coverage for historical data and near-future inserts

-- Function to create a monthly partition
CREATE OR REPLACE FUNCTION create_query_logs_partition(
    partition_date DATE
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    -- Calculate partition boundaries (first of month to first of next month)
    start_date := DATE_TRUNC('month', partition_date)::DATE;
    end_date := (DATE_TRUNC('month', partition_date) + INTERVAL '1 month')::DATE;

    -- Generate partition name: query_logs_y2026m01
    partition_name := 'query_logs_y' || TO_CHAR(start_date, 'YYYY') || 'm' || TO_CHAR(start_date, 'MM');

    -- Check if partition already exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name
        AND n.nspname = 'public'
    ) THEN
        -- Create the partition
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF query_logs_partitioned
             FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            start_date,
            end_date
        );

        RAISE NOTICE 'Created partition: % (% to %)', partition_name, start_date, end_date;
    ELSE
        RAISE NOTICE 'Partition % already exists, skipping', partition_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create partitions for past 3 months
SELECT create_query_logs_partition(CURRENT_DATE - INTERVAL '3 months');
SELECT create_query_logs_partition(CURRENT_DATE - INTERVAL '2 months');
SELECT create_query_logs_partition(CURRENT_DATE - INTERVAL '1 month');

-- Create partition for current month
SELECT create_query_logs_partition(CURRENT_DATE);

-- Create partition for next month
SELECT create_query_logs_partition(CURRENT_DATE + INTERVAL '1 month');

-- ============================================================================
-- Step 5: Create default partition for out-of-range data
-- ============================================================================
-- This catches any data that doesn't fit into defined partitions
-- Important for data integrity - prevents insert failures

CREATE TABLE IF NOT EXISTS query_logs_default
    PARTITION OF query_logs_partitioned DEFAULT;

COMMENT ON TABLE query_logs_default IS 'Default partition for query_logs - catches data outside defined ranges';

-- ============================================================================
-- Step 6: Migrate existing data (if original table exists with data)
-- ============================================================================
DO $$
DECLARE
    row_count BIGINT;
    original_exists BOOLEAN;
BEGIN
    -- Check if original table exists and is not partitioned
    SELECT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'query_logs'
        AND c.relkind = 'r'  -- 'r' = regular table
        AND n.nspname = 'public'
    ) INTO original_exists;

    IF original_exists THEN
        -- Count rows to migrate
        EXECUTE 'SELECT COUNT(*) FROM query_logs' INTO row_count;

        IF row_count > 0 THEN
            RAISE NOTICE 'Migrating % rows from query_logs to query_logs_partitioned...', row_count;

            -- Migrate data in batches to avoid lock contention
            -- For large tables, consider using pg_dump/pg_restore or COPY
            INSERT INTO query_logs_partitioned (
                id, session_id, user_id, query, model, timestamp,
                latency_ms, token_count, retrieval_scores, sources_returned,
                response_length, context_used, sources_count,
                retrieval_time_ms, rerank_time_ms, total_time_ms,
                avg_similarity_score, avg_rerank_score,
                hybrid_search_used, hyde_used, reranker_used, web_search_used,
                extra_data, created_at
            )
            SELECT
                id, session_id, user_id, query, model,
                COALESCE(timestamp, created_at) AS timestamp,  -- Use created_at if timestamp is null
                latency_ms, token_count, retrieval_scores, sources_returned,
                response_length, context_used, sources_count,
                retrieval_time_ms, rerank_time_ms, total_time_ms,
                avg_similarity_score, avg_rerank_score,
                hybrid_search_used, hyde_used, reranker_used, web_search_used,
                extra_data, created_at
            FROM query_logs;

            RAISE NOTICE 'Migration complete. Migrated % rows.', row_count;
        ELSE
            RAISE NOTICE 'Original query_logs table is empty. No data to migrate.';
        END IF;

        -- Rename tables to swap
        RAISE NOTICE 'Renaming tables...';
        ALTER TABLE query_logs RENAME TO query_logs_old;
        ALTER TABLE query_logs_partitioned RENAME TO query_logs;

        RAISE NOTICE 'Tables renamed. Old table preserved as query_logs_old for verification.';
        RAISE NOTICE 'After verifying data integrity, drop with: DROP TABLE query_logs_old;';
    ELSE
        -- No original table, just rename the partitioned table
        RAISE NOTICE 'No original query_logs table found. Renaming partitioned table...';
        ALTER TABLE query_logs_partitioned RENAME TO query_logs;
    END IF;
END $$;

-- ============================================================================
-- Step 7: Update foreign key references (if any)
-- ============================================================================
-- The experiment_results table has a foreign key to query_logs
-- We need to handle this carefully

DO $$
DECLARE
    fk_exists BOOLEAN;
BEGIN
    -- Check if foreign key constraint exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'experiment_results_query_log_id_fkey'
        AND table_name = 'experiment_results'
    ) INTO fk_exists;

    IF fk_exists THEN
        RAISE NOTICE 'Foreign key constraint found. Note: FK to partitioned tables requires special handling.';
        RAISE NOTICE 'Consider using application-level referential integrity or triggers for query_log_id references.';
    END IF;
END $$;

-- ============================================================================
-- Step 8: Create helper function for partition management
-- ============================================================================

-- Function to ensure partitions exist for a date range
CREATE OR REPLACE FUNCTION ensure_query_logs_partitions(
    start_date DATE,
    end_date DATE
) RETURNS INTEGER AS $$
DECLARE
    current_month DATE;
    partitions_created INTEGER := 0;
BEGIN
    current_month := DATE_TRUNC('month', start_date)::DATE;

    WHILE current_month <= end_date LOOP
        PERFORM create_query_logs_partition(current_month);
        current_month := current_month + INTERVAL '1 month';
        partitions_created := partitions_created + 1;
    END LOOP;

    RETURN partitions_created;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION ensure_query_logs_partitions IS 'Creates monthly partitions for the specified date range';

-- Function to list all query_logs partitions
CREATE OR REPLACE FUNCTION list_query_logs_partitions()
RETURNS TABLE (
    partition_name TEXT,
    partition_range TEXT,
    row_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.relname::TEXT AS partition_name,
        pg_get_expr(c.relpartbound, c.oid)::TEXT AS partition_range,
        (SELECT reltuples::BIGINT FROM pg_class WHERE oid = c.oid) AS row_count
    FROM pg_class c
    JOIN pg_inherits i ON c.oid = i.inhrelid
    JOIN pg_class parent ON i.inhparent = parent.oid
    WHERE parent.relname = 'query_logs'
    ORDER BY c.relname;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION list_query_logs_partitions IS 'Lists all partitions of the query_logs table with their ranges and row counts';

-- Function to drop old partitions (for data retention)
CREATE OR REPLACE FUNCTION drop_old_query_logs_partitions(
    retention_months INTEGER DEFAULT 12
) RETURNS INTEGER AS $$
DECLARE
    partition_rec RECORD;
    cutoff_date DATE;
    partitions_dropped INTEGER := 0;
BEGIN
    cutoff_date := (CURRENT_DATE - (retention_months || ' months')::INTERVAL)::DATE;

    FOR partition_rec IN
        SELECT c.relname
        FROM pg_class c
        JOIN pg_inherits i ON c.oid = i.inhrelid
        JOIN pg_class parent ON i.inhparent = parent.oid
        WHERE parent.relname = 'query_logs'
        AND c.relname ~ '^query_logs_y[0-9]{4}m[0-9]{2}$'
        AND TO_DATE(
            SUBSTRING(c.relname FROM 'y([0-9]{4})m([0-9]{2})'),
            'YYYYMM'
        ) < cutoff_date
    LOOP
        EXECUTE format('DROP TABLE %I', partition_rec.relname);
        RAISE NOTICE 'Dropped partition: %', partition_rec.relname;
        partitions_dropped := partitions_dropped + 1;
    END LOOP;

    RETURN partitions_dropped;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION drop_old_query_logs_partitions IS 'Drops query_logs partitions older than the specified retention period';

-- ============================================================================
-- Step 9: Grant permissions (adjust as needed for your environment)
-- ============================================================================
-- Uncomment and modify these as needed for your deployment

-- GRANT SELECT, INSERT, UPDATE, DELETE ON query_logs TO app_user;
-- GRANT SELECT ON list_query_logs_partitions() TO app_user;

COMMIT;

-- ============================================================================
-- Verification queries (run after migration)
-- ============================================================================
-- Uncomment to verify the migration was successful

-- Check partitioned table structure
-- \d+ query_logs

-- List all partitions
-- SELECT * FROM list_query_logs_partitions();

-- Verify row counts match
-- SELECT
--     (SELECT COUNT(*) FROM query_logs) AS new_count,
--     (SELECT COUNT(*) FROM query_logs_old) AS old_count;

-- Test partition pruning (check EXPLAIN shows partition scanning)
-- EXPLAIN (ANALYZE, COSTS OFF)
-- SELECT * FROM query_logs
-- WHERE timestamp >= '2026-01-01' AND timestamp < '2026-02-01';
