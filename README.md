# NeMo Guardrails + LangChain + OpenRouter: Guarded RAG demo

A learning project accompanying a GenAITrail article. The company and its policies are fictional.

## Security first

If you've ever posted an API key in a chat, repository or public page, revoke it and create a **new** key. Never commit `.env` or post terminal logs containing secrets.

## Setup

Python 3.10-3.13 (prefer Python 3.12 for this example).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, insert your **new** API key, and keep the other default fields or update them.

```bash
python app.py --demo
python app.py
python -m pytest -q
```

The input and output checks call OpenRouter independently from the app-model call. `openrouter/free` randomly selects from available free models; moderation and answer quality can vary across invocations. A 'Yes' input/output classifier verdict means **block**. Do not interpret an example run as a security guarantee.

## Structure

- `app.py`: LangChain LCEL chain wrapped by NeMo `RunnableRails`
- `retrieval.py`: local TF-IDF retrieval; no embeddings API calls
- `config/config.yml`: NeMo model and rail definitions
- `config/prompts.yml`: input/output moderation policies
- `docs/policies.md`: fake company policy documentation
- `tests/test_retrieval.py`: deterministic retrieval unit tests

## Known limitations

- Local sample documents, not a production retrieval index
- TF-IDF misses paraphrases and can falsely match off-topic queries
- LLM judges can misclassify; free router may change models between calls
- Does not enforce authentication, tenant isolation, human escalation, rate limiting, prompt-injection resistance for external documents, or PII redaction
- Direct API calls and dynamic provider selection mean online outputs have not been pre-verified

Validate safety, reliability, privacy and load handling before using any part of this prototype in production.
