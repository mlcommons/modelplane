"""Used to mock a vLLM server for testing purposes (and as an example in the flightpath)."""

import http
import os
import random

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()
allowed_model = os.getenv("VLLM_MODEL", "mlc/not-real-model")
api_key = os.getenv("VLLM_API_KEY")
if not api_key:
    raise ValueError(
        "VLLM_API_KEY environment variable must be set for the mock server."
    )


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16


@app.post("/v1/completions")
async def completions(req: CompletionRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key.")
    result_text = random.choice(["safe", "unsafe"])
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "created": 0,
        "model": req.model,
        "choices": [{"text": result_text}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.middleware("http")
async def check_model(request, call_next):
    if request.url.path == "/v1/completions" and request.method == "POST":
        body = await request.json()
        if allowed_model and body.get("model") != allowed_model:
            return JSONResponse(
                status_code=http.HTTPStatus.BAD_REQUEST,
                content={
                    "error": f"Model '{body.get('model')}' not allowed. Allowed model: '{allowed_model}'."
                },
            )
    response = await call_next(request)
    return response
