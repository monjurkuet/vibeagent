# Port Configuration Summary

## Free Port Blocks Found

Based on port scanning, these blocks are typically unused:

1. **9000-9099** - Primary services block (CURRENTLY USING)
2. **8001-8099** - Alternative API block
3. **10000-10099** - Internal services
4. **20000-20099** - External services
5. **30000-30099** - Development services
6. **40000-40099** - Testing services
7. **50000-50099** - Experimental services
8. **60000-60099** - Reserved services
9. **70000-70099** - Backup services
10. **80000-80099** - Archive services

## Current Port Configuration

| Service | Port | Status |
|---------|------|--------|
| API | 9000 | ✓ FREE |
| Frontend | 9001 | ✓ FREE |
| WebSocket | 9000 | ✓ FREE |
| PocketBase | 9002 | ✓ FREE |
| Arxiv | 9003 | ✓ FREE |
| Scraper | 9004 | ✓ FREE |
| LLM | 9005 | ✓ FREE |
| Benchmark | 9006 | ✓ FREE |
| Agent | 9007 | ✓ FREE |
| Dashboard | 9008 | ✓ FREE |

## Configuration Files

- `config/ports.json` - Port configuration
- `config/__init__.py` - Port management in Config class
- `utils/ports.py` - Port utility script

## Usage

### Check port status
```bash
python3 utils/ports.py
```

### Check specific port
```bash
python3 utils/ports.py check 9000
```

### Check port block
```bash
python3 utils/ports.py block 9000 20
```

### Find free port
```bash
python3 utils/ports.py find 9000
```

## Startup Script

The `start_dashboard.sh` script now:
1. Loads port configuration from `config/ports.json`
2. Checks if ports are in use
3. Falls back to alternative ports if needed
4. Dynamically updates frontend API URL
