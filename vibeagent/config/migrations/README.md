# Database Migrations

This directory contains versioned database migration files for VibeAgent.

## Naming Convention

Migration files must follow the format: `YYYYMMDDHHMMSS_description.sql`

- `YYYYMMDDHHMMSS`: 14-digit timestamp (year, month, day, hour, minute, second)
- `description`: Snake_case description of the migration

Example: `20240124120000_add_paper_tags_table.sql`

## Migration File Structure

Each migration file should contain:

1. **UP migration SQL**: The SQL to apply the migration
2. **ROLLBACK SQL (optional)**: SQL to undo the migration, marked with `-- ROLLBACK:`

### Template

```sql
-- Migration: Add new feature
-- Created: 2024-01-24T12:00:00
-- Description: Brief description of what this migration does

-- UP migration SQL
-- Add your migration SQL here
-- Example:
ALTER TABLE papers ADD COLUMN new_column TEXT;

-- ROLLBACK SQL (optional)
-- Add SQL to rollback this migration below
-- ROLLBACK: ALTER TABLE papers DROP COLUMN new_column;
```

## Creating a New Migration

Use the migration script to create a new migration file:

```bash
python scripts/migrate_db.py create "add paper tags table"
```

This will create a new migration file with the current timestamp and a template.

## Applying Migrations

Apply all pending migrations:

```bash
python scripts/migrate_db.py migrate
```

Apply migrations up to a specific version:

```bash
python scripts/migrate_db.py migrate --to 20240124120000
```

## Rolling Back Migrations

Rollback the last migration:

```bash
python scripts/migrate_db.py rollback
```

Rollback multiple migrations:

```bash
python scripts/migrate_db.py rollback --steps 3
```

Rollback to a specific version:

```bash
python scripts/migrate_db.py rollback --to 20240123120000
```

## Checking Migration Status

View migration status:

```bash
python scripts/migrate_db.py status
```

List all migrations:

```bash
python scripts/migrate_db.py list
```

## Validating Migrations

Validate all migration files:

```bash
python scripts/migrate_db.py validate
```

## Best Practices

1. **Always write rollback SQL**: Make sure migrations can be safely rolled back
2. **Keep migrations idempotent**: Migrations should be safe to run multiple times
3. **Test migrations**: Always test migrations on a copy of production data
4. **Use transactions**: Wrap migrations in transactions where possible
5. **Be descriptive**: Use clear, descriptive names for migrations
6. **One change per migration**: Keep each migration focused on a single change
7. **Don't modify existing migrations**: Create new migrations instead

## Migration History

All applied migrations are tracked in the `schema_migrations` table in the database.

| Column | Type | Description |
|--------|------|-------------|
| version | TEXT (PRIMARY KEY) | Migration timestamp |
| name | TEXT | Migration name |
| applied_at | TIMESTAMP | When the migration was applied |
| rollback_sql | TEXT | SQL to rollback the migration |
| checksum | TEXT | MD5 checksum of the migration file |
