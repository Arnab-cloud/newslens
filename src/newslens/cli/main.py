from pathlib import Path

import typer

from newslens.core.engine import InferenceEngine
from newslens.core.model import NewsLens
from newslens.training.finetuner import NewsLensFinetuner

app = typer.Typer(help="NewsLens: The perspective-aligned summarization framework.")


@app.command()
def serve(
    mode: str = typer.Option(
        "a2a", help="Mode to run: 'a2a' (REST API) or 'mcp' (MCP Server)"
    ),
    port: int = typer.Option(8000, help="Port to run the service on"),
    adapter: str = typer.Option("./my_adapter", help="Path to custom LoRA adapter"),
):

    engine = InferenceEngine(adapter_path=adapter)
    lens = NewsLens(engine=engine)

    """Start the NewsLens inference service."""
    if mode == "a2a":
        import uvicorn

        from newslens.server.api import app as fastapi_app
        from newslens.server.api import lens_storage

        lens_storage["model"] = lens

        print(f"🚀 Starting A2A server on port {port}...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
    elif mode == "mcp":
        from newslens.agents.mcp_server import create_mcp_server

        print("🔗 Starting MCP server (stdio)...")
        server = create_mcp_server(lens)
        server.run(transport="stdio")


@app.command()
def train(
    train_data: Path = typer.Argument(..., help="Path to train.csv"),
    val_data: Path = typer.Argument(..., help="Path to val.csv"),
    output_dir: str = typer.Option(
        "./adapter_output", help="Where to save the LoRA weights"
    ),
):
    """Fine-tune the model with custom news data."""
    print(f"🏋️ Starting training on {train_data}...")
    tuner = NewsLensFinetuner()
    tuner.train(str(train_data), str(val_data), output_dir=output_dir)


if __name__ == "__main__":
    app()
