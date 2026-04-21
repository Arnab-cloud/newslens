### File Structure

```
newslens/
├── src/
│ └── newslens/
│ ├── __init__.py # Package entry point
│ ├── core/ # The "Engine Room"
│ │ ├── engine.py # Logic for loading Qwen + LoRA adapters
│ │ ├── model.py # The main NewsLens class (Sync/Async API)
│ │ └── config.py # Default hyperparameters & model paths
│ ├── agents/ # Framework adapters
│ │ ├── langchain.py # Tools for LangGraph/LangChain
│ │ ├── autogen.py # Skill definitions for AutoGen
│ │ └── mcp.py # Model Context Protocol (MCP) server
│ ├── server/ # Connectivity Layer
│ │ ├── api.py # FastAPI A2A (Agent-to-Agent) endpoints
│ │ └── schemas.py # Pydantic data models for payloads
│ ├── training/ # Finetuning module
│ │ └── finetuner.py # Your training logic, modularized
│ ├── cli/ # Command-line interface
│ │ └── main.py # Entry point for 'newslens' commands
│ └── utils/ # Helpers (Batching, token counting)
├── tests/ # Unit and integration tests
├── examples/ # Jupyter notebooks & demo scripts
├── pyproject.toml # Modern dependency & build management
└── README.md # Documentation
```
