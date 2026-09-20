"""Enterprise surfaces: Claude on AWS Bedrock, GPT on Azure OpenAI, Gemini on Vertex AI.

What these tests verify, stated exactly: the REAL SDK client classes are constructed (offline,
with placeholder credentials in the environment), the surface's identifier -- Bedrock model id,
Azure deployment name, Vertex model name -- is what the request carries while the validated
profile name is what the gate and price table key on, a missing configuration fails before any
client exists, and the public surface is named as such in the warnings. Only the request method is
replaced, so a constructor-signature drift in a vendor SDK fails here rather than at a customer's
first call.

What they do NOT verify: a round trip against a live enterprise account. None was available when
this was written, and the README says so.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cortec import Schema, Generator, GenerationError
from cortec.generate import SURFACES


def _schema() -> Schema:
    return Schema(name="t", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b"]},
                  target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 90]})


def _anthropic_resp(text):
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                              output_tokens_details=SimpleNamespace(thinking_tokens=3)),
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn")


def _openai_resp(text):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=2)),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def _gemini_resp(text):
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5,
                                       thoughts_token_count=4),
        text=text)


CSV = "age,grp,y\n33,a,YES\n41,b,NO"


# ── configuration must fail loudly, before any client exists ──────────────────────────

def test_unknown_surface_is_refused():
    with pytest.raises(GenerationError, match="unknown surface"):
        Generator(_schema(), backend="anthropic", model="claude-fable-5", surface="on-prem")


@pytest.mark.parametrize("surface,vendor", [(s, v) for s, v in SURFACES.items() if v])
def test_surface_and_backend_must_match(surface, vendor, monkeypatch):
    """Bedrock serves Claude, Azure OpenAI serves GPT, Vertex serves Gemini; a mismatch is a
    configuration error, not a request that fails later with a vendor message."""
    other = {"anthropic": ("openai", "gpt-5"), "openai": ("gemini", "gemini-3.5-flash"),
             "gemini": ("anthropic", "claude-fable-5")}[vendor]
    with pytest.raises(GenerationError, match="serves"):
        Generator(_schema(), backend=other[0], model=other[1], surface=surface)


def test_bedrock_needs_a_region(monkeypatch):
    for k in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(GenerationError, match="AWS_REGION"):
        Generator(_schema(), backend="anthropic", model="claude-fable-5", surface="bedrock")


def test_bedrock_without_botocore_fails_before_the_first_request(monkeypatch):
    """The SDK builds AnthropicBedrock without botocore and fails only when it signs the first
    request; the tool checks at construction and names the extra to install."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setitem(sys.modules, "botocore", None)      # makes `import botocore` raise
    with pytest.raises(GenerationError, match=r"cortec\[bedrock\]"):
        Generator(_schema(), backend="anthropic", model="claude-fable-5", surface="bedrock")


def test_azure_needs_endpoint_and_deployment(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(GenerationError, match="AZURE_OPENAI_ENDPOINT"):
        Generator(_schema(), backend="openai", model="gpt-5", surface="azure", surface_model="dep")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    with pytest.raises(GenerationError, match="deployment"):
        Generator(_schema(), backend="openai", model="gpt-5", surface="azure")


def test_vertex_needs_a_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(GenerationError, match="GOOGLE_CLOUD_PROJECT"):
        Generator(_schema(), backend="gemini", model="gemini-3.5-flash", surface="vertex")


# ── the real client classes, with only the transport replaced ──────────────────────────

def test_bedrock_constructs_the_real_client_and_sends_the_bedrock_model_id(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAPLACEHOLDER")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "placeholder")
    monkeypatch.setattr("cortec.generate._require_bedrock_deps", lambda: None)
    g = Generator(_schema(), backend="anthropic", model="claude-fable-5", surface="bedrock",
                  surface_model="anthropic.claude-fable-5-v1:0")
    assert isinstance(g._client, anthropic.AnthropicBedrock)
    seen = {}

    def fake_create(**kw):
        seen.update(kw)
        return _anthropic_resp(CSV)
    monkeypatch.setattr(g._client.messages, "create", fake_create)
    assert g._call("prompt") == CSV
    assert seen["model"] == "anthropic.claude-fable-5-v1:0", "the surface's id must be sent"
    assert g.model == "claude-fable-5", "the profile name is what the gate and prices key on"
    assert seen["output_config"] == {"effort": "high"}, "reasoning control is unchanged by surface"
    assert g.stats.calls_with_reasoning_block == 1 and g.stats.spend_usd > 0
    assert "surface=bedrock" in g.describe_surface()
    assert not any("PUBLIC developer API" in w for w in g.stats.warnings)


def test_azure_constructs_the_real_client_and_addresses_the_deployment(monkeypatch):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "placeholder")
    g = Generator(_schema(), backend="openai", model="gpt-5", surface="azure",
                  surface_model="gpt5-prod-deployment")
    assert isinstance(g._client, openai.AzureOpenAI)
    seen = {}

    def fake_create(**kw):
        seen.update(kw)
        return _openai_resp(CSV)
    monkeypatch.setattr(g._client.chat.completions, "create", fake_create)
    assert g._call("prompt") == CSV
    assert seen["model"] == "gpt5-prod-deployment"
    assert "reasoning_effort" not in seen, "reasoning='on' leaves the vendor default alone"
    assert g.stats.thinking_tokens == 2
    assert not any("PUBLIC developer API" in w for w in g.stats.warnings)


def test_vertex_constructs_the_real_client_in_vertex_mode(monkeypatch):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "example-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    g = Generator(_schema(), backend="gemini", model="gemini-3.5-flash", surface="vertex")
    assert isinstance(g._client, genai.Client)
    assert getattr(g._client, "vertexai", True) is True
    seen = {}

    def fake_generate(**kw):
        seen.update(kw)
        return _gemini_resp(CSV)
    monkeypatch.setattr(g._client.models, "generate_content", fake_generate)
    assert g._call("prompt") == CSV
    assert seen["model"] == "gemini-3.5-flash"
    assert g.stats.thinking_tokens == 4
    assert not any("PUBLIC developer API" in w for w in g.stats.warnings)


def test_public_surface_is_named_in_the_warnings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    g = Generator(_schema(), backend="anthropic", model="claude-fable-5")
    assert g.surface == "public"
    assert any("PUBLIC developer API" in w and "surface='bedrock'" in w for w in g.stats.warnings)
