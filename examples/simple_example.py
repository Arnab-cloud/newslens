from newslens.core.mock_engine import MockEngine
from newslens.core.model import NewsLens

# --- Local Development ---
# from newslens.core.engine import InferenceEngine
# engine = InferenceEngine(adapter_path="./my-loRA")

# --- Using a Remote Server ---
# from newslens.core.remote_engine import RemoteEngine
# engine = RemoteEngine(api_url="https://api.domain.com")

# --- Unit Testing ---
engine = MockEngine()

lens = NewsLens(engine=engine)
summary = lens.summarize("Market news here...")
print(summary)
