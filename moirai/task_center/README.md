# Task Center

Task Center is the core coordination service of the moirai network, responsible for task lifecycle management, miner selection, audit task distribution, and reward calculation.

## Installation

### 1. Install Dependencies

```bash
cd moirai
pip install -r requirements.txt
pip install -e .
```

### 2. Database Setup

```bash
# Create PostgreSQL database
sudo -u postgres psql -c "CREATE USER moirai WITH PASSWORD 'moirai';"
sudo -u postgres psql -c "CREATE DATABASE moirai OWNER moirai;"
```

### 3. Configuration File

```bash
cp task_center/config.example.yml task_center/config.yml
```

Edit `config.yml`:

```yaml
database:
  url: postgresql://moirai:moirai@localhost:5432/moirai

bittensor:
  netuid: 361
  chain_endpoint: wss://test.finney.opentensor.ai:443

logging:
  level: INFO
  file: logs/task_center.log
```

## Starting the Service

### Option 1: Direct Start

```bash
# Run from project root
python -m moirai.task_center.task_center_main
```

### Option 2: Using PM2

```bash
# Start with PM2
pm2 start "python -m moirai.task_center.task_center_main" \
  --name task-center \
  --cwd /opt

# Save PM2 process list
pm2 save

# View logs
pm2 logs moirai-task-center

# Monitor
pm2 monit

# Restart
pm2 restart moirai-task-center

# Stop
pm2 stop moirai-task-center
```

### Option 3: Using uvicorn

```bash
# Note: Must use --loop asyncio because bittensor doesn't support uvloop
uvicorn moirai.task_center.task_center_main:app --host 0.0.0.0 --port 8000 --loop asyncio
```