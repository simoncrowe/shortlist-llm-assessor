# shortlist-llm-assessor

A profile assessment tool that uses [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) to evaluate text profiles against configurable criteria. Profiles that match a positive response pattern are forwarded to a notification endpoint; the rest are logged and discarded.

## How it works

1. Reads a **profile** (JSON with a `text` field) and a **config** (system prompt, access token, regex pattern).
2. Sends the profile text to the LLM as a user message, with the config's system prompt providing assessment instructions.
3. Checks the LLM's response against a configurable regex pattern (`llmPositiveResponseRegex`).
4. If matched, POSTs the full profile to a webhook URL. Otherwise, logs the rejection.

## Configuration

The application is configured via environment variables and a JSON config file.

### Environment variables

| Variable | Description |
|---|---|
| `CACHE_DIR` | Directory for caching the Hugging Face model |
| `PROFILE_PATH` | Path to the profile JSON file |
| `CONFIG_PATH` | Path to the configuration JSON file |
| `NOTIFIER_URL` | Webhook URL to POST accepted profiles to |

### Config file

```json
{
  "accessToken": "hf_...",
  "llmSystemPrompt": "Evaluate the following candidate. Respond with only 'yes' or 'no'.",
  "llmPositiveResponseRegex": "[Yy]es"
}
```

## Docker

```bash
docker build -f docker/Dockerfile -t llm-assessor .

docker run \
  -e CACHE_DIR=/cache \
  -e PROFILE_PATH=/input/profile.json \
  -e CONFIG_PATH=/input/config.json \
  -e NOTIFIER_URL=https://example.com/notify \
  -v ./data:/input \
  -v ./cache:/cache \
  llm-assessor
```

The Docker image uses `pytorch/pytorch` with CUDA 12.4 as its base, so a compatible NVIDIA GPU is expected at runtime.

A CI workflow builds and pushes images to GitHub Container Registry on pushes to `main` and on version tags.

## Development

```bash
poetry install --with=dev
poetry run flake8 --ignore=E501
poetry run mypy .
```

## License

MIT
