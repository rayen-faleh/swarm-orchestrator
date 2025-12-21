# LLMLingua Integration Guide

## Overview

LLMLingua is Microsoft's prompt compression library that removes non-essential tokens while preserving semantic meaning. It achieves up to 20x compression with minimal performance loss.

## How It Works

LLMLingua uses a compact language model (GPT-2 small or LLaMA-based) to identify and remove redundant tokens. The library analyzes token importance based on perplexity and removes low-information tokens while preserving critical content.

**LLMLingua-2** (recommended) uses a trained classifier model that's faster and more accurate for out-of-domain content like code.

## Installation

```
pip install llmlingua
```

Requires: transformers, accelerate, torch, tiktoken, nltk

## Key Parameters

- **rate**: Target compression ratio (e.g., 0.3 = compress to 30% of original)
- **target_token**: Target output token count
- **force_tokens**: List of tokens to always preserve (useful for code markers)
- **use_llmlingua2**: Enable the faster, more accurate v2 compression

## Integration Points in Swarm-Orchestrator

### 1. Diff Compression (Voting Phase)

**Where**: Before sending implementation diffs to agents for comparison voting

**Why**: Diffs can be verbose with unchanged context lines. Compression preserves the actual changes (+/-) while reducing boilerplate.

**Approach**: Compress each agent's diff before including in voting prompts. Force-preserve diff markers (+, -, @@) and code keywords (def, class, function).

**Expected savings**: 40-60% token reduction on typical diffs

### 2. Exploration Context

**Where**: After the exploration phase gathers codebase context

**Why**: Exploration often retrieves more context than needed. Compression removes redundant file headers, imports, and boilerplate.

**Approach**: Compress the aggregated exploration results before passing to decomposition. Set target token count based on task complexity.

**Expected savings**: 30-50% token reduction

### 3. MCP Tool Outputs

**Where**: When MCP tools return large outputs (file contents, search results)

**Why**: Tool outputs often include full file contents when only relevant sections matter.

**Approach**: Compress tool outputs before adding to agent context. Preserve function signatures and key identifiers.

**Expected savings**: 50-70% on large file reads

### 4. Agent Prompt Context

**Where**: The shared context section of agent prompts

**Why**: Each of the 5 agents receives similar context. Compressing shared sections reduces per-agent cost.

**Approach**: Compress the task description and codebase context portions. Keep implementation instructions uncompressed for clarity.

**Expected savings**: 20-30% on prompt construction

## Considerations

- **Minimum size threshold**: Skip compression for content under 500 tokens (overhead exceeds benefit)
- **Code preservation**: Always force-preserve language-specific keywords and syntax markers
- **Quality tradeoff**: More aggressive compression (rate < 0.2) risks losing important details
- **GPU requirement**: LLMLingua models run on GPU; CPU fallback is slower but functional

## Model Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| General text | microsoft/llmlingua-2-xlm-roberta-large-meetingbank |
| Code-heavy | microsoft/phi-2 |
| Low memory (<8GB) | TheBloke/Llama-2-7b-Chat-GPTQ |

## References

- [LLMLingua Official Site](https://www.llmlingua.com/)
- [GitHub Repository](https://github.com/microsoft/LLMLingua)
- [PyPI Package](https://pypi.org/project/llmlingua/)
- [LangChain Integration](https://python.langchain.com/docs/integrations/retrievers/llmlingua/)
