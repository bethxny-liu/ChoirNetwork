"""FastAPI interface for the evaluated hymn search pipeline."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from choirnetwork.engine import load_engine
from choirnetwork.scraper import HYMN_BASE_URL

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=10, ge=1, le=50)


class HymnResult(BaseModel):
    label: str
    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    results: list[HymnResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_path = Path(os.environ.get("CHOIRNETWORK_INDEX_PATH", "data/index"))
    app.state.engine = load_engine(index_path)
    yield
    app.state.engine = None


app = FastAPI(title="ChoirNetwork", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "hymns_indexed": len(app.state.engine.index.slugs)}


@app.post("/api/search", response_model=SearchResponse)
def search_hymns(body: SearchRequest) -> SearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    matches = app.state.engine.search(query, top_k=body.top_k)
    return SearchResponse(
        query=query,
        results=[
            HymnResult(
                label=match.label,
                title=match.title,
                url=f"{HYMN_BASE_URL}/{match.slug}",
                snippet=match.snippet,
            )
            for match in matches
        ],
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
