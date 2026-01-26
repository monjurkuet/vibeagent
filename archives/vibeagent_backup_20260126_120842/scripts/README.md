# Database Scripts

This directory contains scripts for managing the VibeAgent database.

## Scripts

### init_db.py

Database initialization script that creates the database schema and seeds initial data.

**Usage:**

```bash
# Initialize database with schema and seed data
python3 scripts/init_db.py

# Use a custom database path
python3 scripts/init_db.py --db-path /path/to/database.db

# Force re-initialization (deletes existing database)
python3 scripts/init_db.py --force

# Skip seeding default data
python3 scripts/init_db.py --skip-seeds

# Enable verbose logging
python3 scripts/init_db.py -v
```

**What it does:**

1. Creates the data directory if it doesn't exist
2. Initializes the database schema from `config/schema.sql`
3. Seeds default configurations
4. Seeds test cases from `tests/test_cases.py`
5. Seeds default system prompts
6. Verifies database creation

### migrate_db.py

Database migration script for managing versioned schema changes.

**Usage:**

```bash
# Show migration status
python3 scripts/migrate_db.py status

# List all migrations
python3 scripts/migrate_db.py list

# Apply pending migrations
python3 scripts/migrate_db.py migrate

# Apply migrations up to a specific version
python3 scripts/migrate_db.py migrate --to 20240124120000

# Rollback the last migration
python3 scripts/migrate_db.py rollback

# Rollback multiple migrations
python3 scripts/migrate_db.py rollback --steps 3

# Rollback to a specific version
python3 scripts/migrate_db.py rollback --to 20240123120000

# Validate all migration files
python3 scripts/migrate_db.py validate

# Create a new migration
python3 scripts/migrate_db.py create "add new feature"

# Use custom paths
python3 scripts/migrate_db.py --db-path /path/to/db.db --migrations-dir /path/to/migrations status
```

**Commands:**

- `migrate` - Apply pending migrations
- `rollback` - Rollback migrations
- `status` - Show migration status
- `list` - List all migrations
- `validate` - Validate migration files
- `create` - Create a new migration file

## Database Schema

The database contains the following tables:

- **papers** - Stores arXiv research papers
- **prompts** - Stores system prompts
- **test_cases** - Stores test scenario configurations
- **configurations** - Stores application settings
- **agent_interactions** - Logs agent operations
- **schema_migrations** - Tracks migration history

## Best Practices

1. **Always backup before migrations**: Backup your database before running migrations
2. **Test migrations**: Test migrations on a development database first
3. **Use transactions**: Migrations are wrapped in transactions for safety
4. **Write rollback SQL**: Always include rollback SQL in migration files
5. **Version control**: Keep migration files under version control

## Example Workflow

### Initial Setup

```bash
# Initialize the database
python3 scripts/init_db.py

# Check migration status
python3 scripts/migrate_db.py status
```

### Making Schema Changes

```bash
# Create a new migration
python3 scripts/migrate_db.py create "add user preferences table"

# Edit the generated migration file
# Add your UP and ROLLBACK SQL

# Validate the migration
python3 scripts/migrate_db.py validate

# Apply the migration
python3 scripts/migrate_db.py migrate
```

### Rolling Back Changes

```bash
# Rollback the last migration
python3 scripts/migrate_db.py rollback

# Check the status
python3 scripts/migrate_db.py status
```

## Troubleshooting

### Migration Fails

If a migration fails:

1. Check the error message
2. Fix the issue in the migration file
3. Manually rollback if needed
4. Validate the migration file
5. Try again

### Database Already Exists

```bash
# Force re-initialization (WARNING: deletes all data)
python3 scripts/init_db.py --force
```

### Schema Mismatch

```bash
# Check what migrations are pending
python3 scripts/migrate_db.py list

# Validate migration files
python3 scripts/migrate_db.py validate

# Apply pending migrations
python3 scripts/migrate_db.py migrate
```