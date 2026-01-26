-- Migration: Add paper tags table
-- Created: 2024-01-24T12:00:00
-- Description: Creates a tags table and many-to-many relationship with papers

-- Create tags table
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    color TEXT DEFAULT '#007bff',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create junction table for paper-tag relationships
CREATE TABLE IF NOT EXISTS paper_tags (
    paper_arxiv_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (paper_arxiv_id, tag_id),
    FOREIGN KEY (paper_arxiv_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_paper_tags_tag_id ON paper_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

-- ROLLBACK SQL
-- ROLLBACK: 
-- DROP INDEX IF EXISTS idx_tags_name;
-- DROP INDEX IF EXISTS idx_paper_tags_tag_id;
-- DROP TABLE IF EXISTS paper_tags;
-- DROP TABLE IF EXISTS tags;