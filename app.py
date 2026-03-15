"""
Streamlit web application for benchmarking Toon format vs JSON format with LLMs.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from src.benchmark import BUILTIN_SCENARIOS, BenchmarkRun, run_benchmark
from src.llm_client import LLMClient
from src.toon_format import dumps as to_toon, loads as from_toon

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Toon vs JSON — LLM Benchmark",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI (or compatible) API key.",
    )
    base_url = st.text_input(
        "API Base URL (optional)",
        value=os.getenv("OPENAI_BASE_URL", ""),
        help="Leave blank for the default OpenAI endpoint.",
    )
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_tokens = st.number_input("Max output tokens", min_value=64, max_value=4096, value=512, step=64)

    st.markdown("---")
    st.caption("Toon Format: Token-Optimized Object Notation")
    st.caption("Keys are unquoted · T/F for booleans · ~ for null")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("⚡ Toon Format vs JSON — LLM Benchmark")
st.markdown(
    """
Compare how an LLM performs when the **same structured data** is provided in
**JSON** versus **Toon** format.  The benchmark measures:

| Metric | Description |
|--------|-------------|
| 🔢 Token count | Fewer tokens → lower cost |
| 💵 Cost (USD) | Estimated cost per call |
| ⏱️ Latency | Time to receive the response |
| 📝 Output quality | Side-by-side output comparison |
"""
)

tab_benchmark, tab_converter, tab_about = st.tabs(
    ["🏁 Run Benchmark", "🔄 Format Converter", "ℹ️ About Toon Format"]
)

# ===========================================================================
# TAB 1 — Run Benchmark
# ===========================================================================
with tab_benchmark:
    st.subheader("Select Scenarios")
    scenario_names = [s.name for s in BUILTIN_SCENARIOS]
    selected_names = st.multiselect(
        "Scenarios to run",
        options=scenario_names,
        default=scenario_names[:2],
        help="Each scenario will be run once with JSON input and once with Toon input.",
    )

    run_button = st.button("▶️ Run Benchmark", type="primary", disabled=not api_key)

    if not api_key:
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to run the benchmark.")

    if run_button and selected_names:
        client = LLMClient(
            api_key=api_key,
            base_url=base_url or None,
            model=model,
        )
        scenarios = [s for s in BUILTIN_SCENARIOS if s.name in selected_names]
        runs: list[BenchmarkRun] = []

        progress = st.progress(0, text="Running benchmark…")
        for i, scenario in enumerate(scenarios):
            with st.spinner(f"Running: {scenario.name}…"):
                try:
                    run = run_benchmark(
                        scenario,
                        client,
                        temperature=temperature,
                        max_tokens=int(max_tokens),
                    )
                    runs.append(run)
                except Exception as exc:
                    st.error(f"Error in scenario '{scenario.name}': {exc}")
            progress.progress((i + 1) / len(scenarios), text=f"Completed: {scenario.name}")

        if runs:
            st.success(f"✅ Benchmark complete — {len(runs)} scenario(s) ran.")

            # ---------------------------------------------------------------
            # Aggregate summary table
            # ---------------------------------------------------------------
            st.subheader("📊 Summary")
            summary_rows = []
            for run in runs:
                j = run.json_result.response
                t = run.toon_result.response
                summary_rows.append(
                    {
                        "Scenario": run.scenario_name,
                        "JSON input tokens": j.input_tokens,
                        "Toon input tokens": t.input_tokens,
                        "Token savings": run.input_token_savings,
                        "Token savings %": f"{run.input_token_savings_pct:.1f}%",
                        "JSON cost ($)": f"{j.total_cost_usd:.6f}",
                        "Toon cost ($)": f"{t.total_cost_usd:.6f}",
                        "Cost savings ($)": f"{run.cost_savings_usd:.6f}",
                        "JSON latency (s)": f"{j.latency_seconds:.2f}",
                        "Toon latency (s)": f"{t.latency_seconds:.2f}",
                        "Latency diff (s)": f"{run.latency_diff_seconds:+.2f}",
                    }
                )
            st.dataframe(summary_rows, use_container_width=True)

            # ---------------------------------------------------------------
            # Per-scenario detail
            # ---------------------------------------------------------------
            st.subheader("🔍 Per-Scenario Details")
            for run in runs:
                with st.expander(f"📋 {run.scenario_name}", expanded=True):
                    col_json, col_toon = st.columns(2)

                    with col_json:
                        st.markdown("### JSON Input")
                        st.code(run.json_result.input_text, language="json")
                        st.metric("Input tokens", run.json_result.response.input_tokens)
                        st.metric("Output tokens", run.json_result.response.output_tokens)
                        st.metric("Cost (USD)", f"${run.json_result.response.total_cost_usd:.6f}")
                        st.metric("Latency (s)", f"{run.json_result.response.latency_seconds:.2f}")
                        st.markdown("**LLM Output:**")
                        st.text_area(
                            "JSON output",
                            value=run.json_result.response.content,
                            height=200,
                            key=f"json_out_{run.scenario_name}",
                        )

                    with col_toon:
                        st.markdown("### Toon Input")
                        st.code(run.toon_result.input_text, language="text")
                        delta_tokens = run.input_token_savings
                        delta_pct = run.input_token_savings_pct
                        st.metric(
                            "Input tokens",
                            run.toon_result.response.input_tokens,
                            delta=f"-{delta_tokens} ({delta_pct:.1f}%)",
                            delta_color="inverse",
                        )
                        st.metric("Output tokens", run.toon_result.response.output_tokens)
                        st.metric(
                            "Cost (USD)",
                            f"${run.toon_result.response.total_cost_usd:.6f}",
                            delta=f"-${run.cost_savings_usd:.6f}",
                            delta_color="inverse",
                        )
                        st.metric(
                            "Latency (s)",
                            f"{run.toon_result.response.latency_seconds:.2f}",
                            delta=f"{run.latency_diff_seconds:+.2f}",
                            delta_color="inverse",
                        )
                        st.markdown("**LLM Output:**")
                        st.text_area(
                            "Toon output",
                            value=run.toon_result.response.content,
                            height=200,
                            key=f"toon_out_{run.scenario_name}",
                        )


# ===========================================================================
# TAB 2 — Format Converter
# ===========================================================================
with tab_converter:
    st.subheader("🔄 Convert Between JSON ↔ Toon")
    st.markdown(
        "Use this converter to explore how your own data looks in Toon format "
        "and understand the token reduction."
    )

    direction = st.radio(
        "Conversion direction",
        ["JSON → Toon", "Toon → JSON"],
        horizontal=True,
    )

    default_json = json.dumps(
        {
            "user": {
                "id": 42,
                "name": "Alice Johnson",
                "active": True,
                "roles": ["admin", "editor"],
                "address": {"city": "New York", "country": "USA"},
            }
        },
        indent=2,
    )

    input_label = "JSON input" if direction == "JSON → Toon" else "Toon input"
    default_input = default_json if direction == "JSON → Toon" else to_toon(
        {
            "user": {
                "id": 42,
                "name": "Alice Johnson",
                "active": True,
                "roles": ["admin", "editor"],
                "address": {"city": "New York", "country": "USA"},
            }
        }
    )

    input_text = st.text_area(input_label, value=default_input, height=250)

    if st.button("Convert"):
        try:
            if direction == "JSON → Toon":
                parsed = json.loads(input_text)
                output = to_toon(parsed)
                output_lang = "text"
            else:
                parsed = from_toon(input_text)
                output = json.dumps(parsed, indent=2)
                output_lang = "json"

            st.markdown("**Output:**")
            st.code(output, language=output_lang)

            # Token comparison
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                in_tokens = len(enc.encode(input_text))
                out_tokens = len(enc.encode(output))
                diff = in_tokens - out_tokens
                pct = diff / in_tokens * 100 if in_tokens else 0
                col1, col2, col3 = st.columns(3)
                col1.metric("Input tokens", in_tokens)
                col2.metric("Output tokens", out_tokens)
                col3.metric(
                    "Token difference",
                    diff,
                    delta=f"{pct:.1f}%",
                    delta_color="inverse" if direction == "JSON → Toon" else "normal",
                )
            except ImportError:
                pass

        except Exception as exc:
            st.error(f"Conversion error: {exc}")


# ===========================================================================
# TAB 3 — About Toon Format
# ===========================================================================
with tab_about:
    st.subheader("ℹ️ What is Toon Format?")
    st.markdown(
        """
**Toon** (Token-Optimized Object Notation) is a compact serialization format
designed to reduce the number of tokens consumed when passing structured data
to and from Large Language Models.

### Key differences from JSON

| Feature | JSON | Toon |
|---------|------|------|
| Object keys | Always quoted | Unquoted when safe |
| Boolean true | `true` | `T` |
| Boolean false | `false` | `F` |
| Null | `null` | `~` |
| Whitespace | Optional but often added | None |
| Strings | Always double-quoted | Unquoted when no special chars |

### Examples

```
# JSON
{"name": "Alice", "active": true, "tags": ["admin", "editor"], "score": null}

# Toon
{name:Alice,active:T,tags:[admin,editor],score:~}
```

### Why does it matter?

LLMs are billed by token.  A typical JSON payload with whitespace and quoted
keys can be 30–50 % larger in token count than the equivalent Toon payload.
Reducing input tokens lowers cost and can also speed up inference (fewer
tokens to process in the attention mechanism).

### Round-trip guarantee

Toon is fully round-trippable:

```python
from src.toon_format import dumps, loads
import json

data = {"active": True, "count": 3}
assert loads(dumps(data)) == data
assert json.loads(json.dumps(data)) == data
```

### When NOT to use Toon

* When the model has been fine-tuned exclusively on JSON.
* When human readability of the raw payload is critical.
* For very simple payloads where the overhead is negligible.
"""
    )
