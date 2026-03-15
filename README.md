# toon-vs-json

A benchmarking application that compares **Toon format** versus **JSON format** as input/output for Large Language Models (LLMs).

## What is measured?

| Metric | Description |
|--------|-------------|
| 🔢 **Token count** | Fewer input tokens → lower cost |
| 💵 **Cost (USD)** | Estimated cost per API call |
| ⏱️ **Latency** | Time to receive the full response |
| 📝 **Output quality** | Side-by-side comparison of LLM outputs |

## What is Toon Format?

**Toon** (Token-Optimized Object Notation) is a compact serialization format designed to reduce token usage when passing structured data to/from LLMs.

### Differences from JSON

| Feature | JSON | Toon |
|---------|------|------|
| Object keys | Always quoted | Unquoted when safe (word chars) |
| Boolean true | `true` | `T` |
| Boolean false | `false` | `F` |
| Null | `null` | `~` |
| Whitespace | Optional (often added) | None |
| Simple strings | Always double-quoted | Unquoted when identifier-safe |

### Example

```
# JSON  (92 chars)
{"name": "Alice", "active": true, "roles": ["admin", "editor"], "score": null}

# Toon  (52 chars — 43% shorter)
{name:Alice,active:T,roles:[admin,editor],score:~}
```

## Project Structure

```
toon-vs-json/
├── app.py                  # Streamlit web UI
├── requirements.txt
├── .env.example
├── src/
│   ├── toon_format.py      # Toon encoder / decoder
│   ├── llm_client.py       # OpenAI-compatible LLM client
│   └── benchmark.py        # Scenarios & orchestration
└── tests/
    ├── test_toon_format.py
    └── test_benchmark.py
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/juan-campuzano/toon-vs-json.git
cd toon-vs-json
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

## Usage

The app has three tabs:

### 🏁 Run Benchmark
Select one or more built-in scenarios and click **▶️ Run Benchmark**.  
Each scenario is executed twice — once with JSON input and once with Toon input.  
Results show token counts, costs, latencies, and the raw LLM outputs side-by-side.

### 🔄 Format Converter
Paste any JSON and instantly see its Toon equivalent (and vice versa), along with
a token-count comparison.

### ℹ️ About Toon Format
Full specification of the Toon format with examples.

## Built-in Scenarios

| Scenario | Description |
|----------|-------------|
| User Profile Extraction | Extract name, email, country from user records |
| Product Catalog Summary | Summarize a product catalog in one paragraph |
| Sales Report Analysis | Answer business questions from a monthly sales report |
| Config File Transformation | Convert a nested config object to `KEY=VALUE` env vars |

## Running Tests

```bash
pytest tests/ -v
```

## Using Toon Format in Code

```python
from src.toon_format import dumps, loads

data = {"name": "Alice", "active": True, "roles": ["admin", "editor"]}

toon_str = dumps(data)
# → {name:Alice,active:T,roles:[admin,editor]}

restored = loads(toon_str)
assert restored == data  # full round-trip guarantee
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI (or compatible) API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use |
| `OPENAI_BASE_URL` | No | OpenAI default | Custom base URL for OpenAI-compatible providers |
