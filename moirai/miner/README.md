# Miner

Miner is the training execution service of the moirai network, responsible for executing LoRA training tasks, uploading models to HuggingFace, and submitting results for validation.

## Features

- **LoRA Training**: Execute text and image LoRA training tasks
- **Multi-GPU Support**: Parallel training on multiple GPUs
- **Queue Management**: Priority-based task queue system
- **Auto Upload**: Automatic model upload to HuggingFace
- **Local Testing**: Quality check before submission
- **Dataset Submission**: Submit datasets for validation before training

## Training Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Receive Task │────▶│ Submit      │────▶│ Wait for    │
│              │     │ Dataset     │     │ Validation  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Submit      │◀────│ Upload to   │◀────│ Execute     │
│ Result      │     │ HuggingFace │     │ Training    │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Installation

### 1. Install Dependencies

```bash
cd moirai
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration File

```bash
cp miner/config.example.yml miner/config.yml
```

Edit `config.yml`:

```yaml
wallet:
  name: miner
  hotkey: default

bittensor:
  netuid: 361
  chain_endpoint: wss://test.finney.opentensor.ai:443

task_center:
  url: http://207.56.24.100/task-center

miner:
  min_stake: 200.0
  max_queue_size: 100

huggingface:
  token: <YOUR_HF_TOKEN>
  username: <YOUR_HF_USERNAME>

training:
  text:
    base_model: HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
    default_lora_rank: 16
    default_lora_alpha: 32
    default_num_train_epochs: 1
    default_iteration_count: 1000
    default_batch_size: 4
    default_learning_rate: 2e-4
    default_max_length: 512

datasets:
  text:
    repository_id: your-username/your-text-dataset
    question_column: question
    answer_column: answer

  image:
    repository_id: your-username/your-image-dataset
    image_column: image
    caption_column: text

logging:
  level: INFO
  file: logs/miner.log
```

## Wallet Configuration

Miners require a Bittensor wallet for authentication, signing submissions, and receiving rewards. The wallet must be registered on the subnet with sufficient stake.

### Step 1: Install Bittensor CLI

```bash
pip install bittensor
```

### Step 2: Create Coldkey

The coldkey is your main wallet that holds your TAO tokens.

```bash
btcli wallet new_coldkey --wallet.name miner
```

You will be prompted to:
1. Enter a password (remember this password!)
2. Save the mnemonic phrase (12 or 24 words) - **KEEP THIS SAFE!**

### Step 3: Create Hotkey

The hotkey is used for day-to-day operations and signing training submissions.

```bash
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

### Step 4: Fund Your Wallet

Transfer TAO to your coldkey address for staking:

```bash
# Get your coldkey address
btcli wallet overview --wallet.name miner
```

### Step 5: Register on Subnet

Register your miner on the moirai subnet (requires TAO for registration fee):

```bash
btcli subnets register --wallet.name miner --wallet.hotkey default --netuid 361 --network test
```

### Step 6: Stake TAO

Stake TAO to meet the minimum stake requirement and earn more rewards:

```bash
# Check current stake
btcli stake list --wallet.name miner

# Add stake (minimum required: 200 TAO)
btcli stake add --wallet.name miner --netuid 361 --amount 200 --network test
```

### Step 7: Verify Registration

```bash
# Check subnet registration
btcli subnets show --netuid 361 --network test

# View your miner info
btcli wallet overview --wallet.name miner
```

### Step 8: Update Configuration

Edit `config.yml` to match your wallet names:

```yaml
wallet:
  name: miner            # Your coldkey name
  hotkey: default        # Your hotkey name
```

### Wallet Directory Structure

```
~/.bittensor/wallets/
└── miner/
    ├── coldkey              # Encrypted coldkey (password protected)
    ├── coldkeypub.txt       # Public coldkey
    └── hotkeys/
        └── default          # Hotkey file
```

## HuggingFace Configuration

Miners need a HuggingFace account to upload trained models.

### Step 1: Create HuggingFace Account

Visit https://huggingface.co and create an account.

### Step 2: Generate Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `moirai-miner`
4. Role: `Write` (required for uploading models)
5. Click "Generate token"
6. Copy the token (starts with `hf_`)

### Step 3: Update Configuration

```yaml
huggingface:
  token: hf_xxxxxxxxxxxxxxxxxxxxx
  username: your-huggingface-username
```

### Step 4: Verify Token

```bash
# Test authentication
huggingface-cli whoami
```

## Starting the Service

### Option 1: Direct Start

```bash
# Run from project root
python -m moirai.miner.miner_main
```

### Option 2: Using PM2

```bash
# Start with PM2
pm2 start "python -m moirai.miner.miner_main" \
  --name moirai-miner \
  --cwd /path/to/moirai-sn361

# Save PM2 process list
pm2 save

# View logs
pm2 logs moirai-miner

# Monitor
pm2 monit

# Restart
pm2 restart moirai-miner

# Stop
pm2 stop moirai-miner
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MINER_CONFIG` | Config file path | `miner/config.yml` |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | All available |
| `HF_TOKEN` | HuggingFace Token | - |
| `HUGGINGFACE_TOKEN` | HuggingFace Token (alternative) | - |

## Recommended Hardware

| Component | Recommended                 |
|-----------|-----------------------------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| CPU | 16 cores                    |
| RAM |  64GB                       |
| Storage | 1TB NVMe SSD               |
| Network | 1Gbps                      |

## Training Configuration

### Text LoRA Parameters

```yaml
training:
  text:
    base_model: HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
    default_lora_rank: 16          # LoRA rank (lower = faster, higher = better quality)
    default_lora_alpha: 32         # LoRA alpha
    default_num_train_epochs: 1    # Number of epochs
    default_iteration_count: 1000  # Max training steps
    default_batch_size: 4          # Batch size
    default_learning_rate: 2e-4    # Learning rate
    default_max_length: 512        # Max sequence length
```

## Network Configuration

```yaml
bittensor:
  netuid: 361
  chain_endpoint: wss://test.finney.opentensor.ai:443

axon:
  enabled: true
  ip: 0.0.0.0
  port: 8001
  external_ip: <YOUR_PUBLIC_IP>
```

> **Important Notice:**
> - Each UID is limited to **one external IP address only**.
> - Please open port **8001** (or the port you configured) on your firewall to ensure your miner can properly receive tasks and allocate resources.
> - Make sure your `external_ip` is set to your public IP address for proper network communication.
