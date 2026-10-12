# RailPulse AI

## Quick reference

- **Source code, Dockerfile, issues:** https://github.com/Siegried81/railpulse-genai
- **Maintained by:** Siegried81
- **Also on:** GitHub Container Registry, `ghcr.io/siegried81/<image>`
- **Architecture:** `linux/amd64`

## Supported tags

- `latest`: the current `main` branch
- `X.Y.Z` and `X`: released versions (git tags `vX.Y.Z`), the ones to pin for a stable deployment
- `sha-xxxxxxx`: one specific commit

## What is it?

Railway operations assistant: ask questions about Belgian rail delays in natural language and get SQL-backed answers and recommendations (DeepSeek, Groq, OpenRouter, Anthropic or Ollama).

Images from this repository:

- `railpulse-genai`: Streamlit UI on 8501

## How to use this image

```bash
docker run -d -p 8501:8501 --env-file .env siegried/railpulse-genai
# open http://localhost:8501
```

## Environment variables

None are required to start the UI. Secrets are **never** baked into the image: pass
them at run time with `-e NAME=value` or `--env-file .env` (see `.env.example` in the repo).

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `deepseek`, `groq`, `openrouter`, `anthropic` or `ollama` |
| `DEEPSEEK_API_KEY / GROQ_API_KEY / ...` | key for the chosen provider |
| `DB_PATH` | SQLite database path, default `data/railpulse_ai.db` |

## Volumes

- `/app/data`: SQLite database
- `/app/reports`: generated weekly reports

## License

See the LICENSE file in the source repository, if any. As with every image, it is
your responsibility to check that your use complies with the licenses of the
software it contains.
