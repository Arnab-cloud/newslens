# NewsLens

NewsLens is a modular Python SDK designed for production-grade news summarization. It provides a unified interface for model inference, supporting local execution, remote API calls, and mock engines for development.

## Installation

NewsLens supports optional dependencies based on your specific requirements.

For standard usage:

```bash
pip install newslens
```

To include specific capabilities, use the following extras:

```bash
# For local GPU inference
pip install "newslens[local]"

# For MCP server support
pip install "newslens[mcp]"

# For all features
pip install "newslens[all]"
```

## Usage

The package utilizes a strategy pattern, allowing you to swap inference backends without modifying your application logic.

### Programmatic API

You can initialize the `NewsLens` class by injecting a specific engine.

```python
from newslens.core.model import NewsLens
from newslens.core.engines import RemoteEngine

# Configure the engine
engine = RemoteEngine(api_url="https://api.yourdomain.com")
lens = NewsLens(engine=engine)

# Summarize a single article
summary = lens.summarize("Your article text here.")

# Summarize multiple articles concurrently
summaries = await lens.asummarize_batch(["Article 1", "Article 2"])
```

### FastAPI Server

The package includes a production-ready API server with lifespan management.

```python
from fastapi import FastAPI
from newslens.server.api import app
# The server automatically manages the thread pool and model lifecycle
```

### Agent Integration

NewsLens can be deployed as an MCP server to provide summarization capabilities to AI agents.

```python
from newslens.agents.mcp import create_mcp_server
from newslens.core.model import NewsLens

lens = NewsLens(engine=my_engine)
mcp_server = create_mcp_server(lens)
mcp_server.run(transport="stdio")
```

## Configuration

The SDK uses `pydantic-settings`. You can configure the `BASE_MODEL`, `ADAPTER_PATH`, and `MAX_WORKERS` via environment variables or a configuration object.

## Testing

The package includes a comprehensive test suite. Ensure you have installed the dev dependencies:

```bash
pip install "newslens[all]"
pytest
```

## License

This project is licensed under the [MIT License](./license)
