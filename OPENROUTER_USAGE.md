# OpenRouter Integration for DP-OPT

This implementation allows you to use OpenRouter API to test multiple LLM models for engineer prompts instead of using local models like `lmsys/vicuna-7b-v1.3`.

## Setup

### 1. Get an OpenRouter API Key

1. Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up and create an API key
3. Set it as an environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Alternatively, you can pass it directly via the `--openrouter_api_key` argument.

### 2. Install Required Dependencies

```bash
pip install requests
```

## Usage

### Basic Usage

To use OpenRouter instead of a local model for engineer prompts:

```bash
python train_opt.py \
  --use_openrouter True \
  --openrouter_model "meta-llama/llama-3-8b-instruct" \
  --model "lmsys/vicuna-7b-v1.3" \
  --data sst2 \
  --ape_mode iid_ibwd \
  --num_prompt 5 \
  --device cuda
```

### Available Models

OpenRouter provides access to many models. Here are some popular options:

**Open Source Models:**
- `meta-llama/llama-3-8b-instruct` (default)
- `meta-llama/llama-3-70b-instruct`
- `mistralai/mistral-7b-instruct`
- `mistralai/mixtral-8x7b-instruct`

**Proprietary Models:**
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-haiku`
- `openai/gpt-4-turbo`
- `openai/gpt-3.5-turbo`
- `google/gemini-pro`

See the full list at [https://openrouter.ai/docs#models](https://openrouter.ai/docs#models)

### Example: Using Claude 3 Haiku (No Local Model)

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

python train_opt.py \
  --use_openrouter True \
  --openrouter_model "anthropic/claude-3-haiku" \
  --data sst2 \
  --ape_mode iid_ibwd \
  --num_prompt 3 \
  --max_new_tokens 100 \
  --gen_temp 0.7 \
  --device cpu
```

**Note:** No `--model` argument needed! OpenRouter handles everything.

### Example: Using GPT-4 Turbo (CPU-Only)

```bash
python train_opt.py \
  --use_openrouter True \
  --openrouter_model "openai/gpt-4-turbo" \
  --openrouter_api_key "sk-or-v1-..." \
  --data trec \
  --ape_mode iid_ibwd \
  --num_prompt 5 \
  --device cpu
```

**Note:** Runs entirely on CPU since no local model inference is performed!

## How It Works

When `--use_openrouter True` is set:

1. **Engineer Prompts:** The backward instruction generator uses the specified OpenRouter model to generate improved prompts based on success/failure examples
2. **Evaluation:** OpenRouter is also used to evaluate the generated prompts - **no local model is needed**
3. **Cost:** You pay for OpenRouter API calls for both prompt generation and evaluation
4. **No Downloads:** No large models are downloaded to your machine

This allows you to:
- **Run without GPU:** No local model means no GPU requirement
- **Test different LLMs:** Try GPT-4, Claude, Llama, etc. without downloading them
- **Low resource usage:** Perfect for laptops or machines without powerful GPUs
- **Quick experimentation:** Start testing immediately without waiting for model downloads

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_openrouter` | False | Enable OpenRouter integration |
| `--openrouter_model` | `meta-llama/llama-3-8b-instruct` | OpenRouter model to use |
| `--openrouter_api_key` | None | API key (or use `OPENROUTER_API_KEY` env var) |
| `--model` | N/A | **Optional when using OpenRouter** - no local model needed |
| `--device` | `cuda` | Use `cpu` when using OpenRouter (no GPU needed) |
| `--gen_temp` | 0.9 | Temperature for prompt generation |
| `--max_new_tokens` | 128 | Max tokens for generated prompts |
| `--rep_penalty` | 1.0 | Repetition penalty |

## Limitations

When using OpenRouter:

- **Ensemble generation** (`--ensemble_gen`) is not fully supported and will be disabled
- **Tokenwise generation** (`--tokenwise_gen`) is not supported and will be disabled
- **DP mechanisms** may have different behavior compared to local models
- **Evaluation is slow**: Makes one API call per sample, which can be slow for large datasets

### ⚠️ Important: Evaluation Speed & Costs

OpenRouter evaluation processes samples sequentially (one API call per sample). For large validation sets:

**Problem:**
- Default SST-2 dataset: ~8,700 samples
- With `--holdout_ratio 0.99`: ~8,600 validation samples
- Time: ~8,600 samples × 2 seconds = **~5 hours**
- Cost: ~$5-10 per evaluation run

**Solutions:**

1. **Use smaller validation set (Recommended):**
   ```bash
   --holdout_ratio 0.01  # Use only 1% for validation (~87 samples)
   ```

2. **Skip evaluation during training:**
   ```bash
   --skip_eval  # Only generate prompts, no evaluation
   ```

3. **Evaluate offline:**
   - Generate prompts with `--skip_eval`
   - Evaluate later with a local model or smaller sample

**Recommended settings for OpenRouter:**
```bash
python train_opt.py \
  --use_openrouter True \
  --openrouter_model "meta-llama/llama-3-8b-instruct" \
  --data sst2 \
  --holdout_ratio 0.01 \
  --num_prompt 3 \
  --device cpu
```

This reduces validation from ~8,600 to ~87 samples (100x faster, 100x cheaper)

## Cost Estimation

OpenRouter charges per token. Approximate costs:

- **Llama 3 8B:** ~$0.10 per 1M tokens
- **Claude 3 Haiku:** ~$0.25 per 1M tokens (input)
- **GPT-3.5 Turbo:** ~$0.50 per 1M tokens (input)
- **GPT-4 Turbo:** ~$10 per 1M tokens (input)

For typical usage with 5 prompts and 128 max tokens, expect:
- Llama 3: < $0.01
- Claude 3 Haiku: < $0.05
- GPT-4: ~$0.50

See [https://openrouter.ai/docs#models](https://openrouter.ai/docs#models) for current pricing.

## Troubleshooting

### API Key Error

```
ValueError: OpenRouter API key must be provided...
```

**Solution:** Set the environment variable or pass `--openrouter_api_key`

### Request Timeout

```
Failed after 3 retries: ...
```

**Solution:** Check your internet connection or try a different model

### Model Not Found

```
OpenRouter API error: model not found
```

**Solution:** Verify the model name at [https://openrouter.ai/docs#models](https://openrouter.ai/docs#models)

## Examples Directory

See `examples/openrouter/` for complete example scripts.
