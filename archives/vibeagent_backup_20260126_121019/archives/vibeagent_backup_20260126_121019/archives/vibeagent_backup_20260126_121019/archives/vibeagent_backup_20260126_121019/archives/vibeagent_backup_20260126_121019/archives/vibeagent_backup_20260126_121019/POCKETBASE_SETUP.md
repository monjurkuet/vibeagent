# PocketBase Setup for VibeAgent

## Quick Start

1. **Download PocketBase:**
   ```bash
   wget https://github.com/pocketbase/pocketbase/releases/download/v0.23.5/pocketbase_0.23.5_linux_amd64.zip
   unzip pocketbase_0.23.5_linux_amd64.zip
   ```

2. **Start PocketBase:**
   ```bash
   ./pocketbase serve --http=0.0.0.0:8090
   ```

3. **Create Collection:**
   - Go to http://localhost:8090/_/
   - Create admin account
   - Create new collection `papers` with fields:
     - `id` (text) - Primary key
     - `title` (text)
     - `authors` (json)
     - `published` (date)
     - `abstract` (text)
     - `summary` (text)
     - `url` (url)
     - `pdf_url` (url)
     - `topics` (json array)

## Usage

Run VibeAgent:
```bash
python vibeagent.py http://localhost:8087/v1 http://localhost:8090 [email] [password]
```

Example with auth:
```bash
python vibeagent.py http://localhost:8087/v1 http://localhost:8090 admin@example.com password123
```

Example without auth:
```bash
python vibeagent.py
```