"""Groq chat-completion client with explicit configuration and API errors."""

from groq import APIError, Groq

from src.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


class MissingGroqAPIKeyError(RuntimeError):
    """Raised when a live response is requested without a configured key."""


class GroqRequestError(RuntimeError):
    """Raised when Groq rejects or cannot complete a request."""


def generate_response(system_prompt: str, user_prompt: str) -> str:
    """Generate a chat response using Groq's OpenAI-compatible API endpoint."""
    if not GROQ_API_KEY:
        raise MissingGroqAPIKeyError(
            "GROQ_API_KEY is missing. Add your key to the local .env file "
            "to enable live answers."
        )
    if not GROQ_MODEL:
        raise RuntimeError("GROQ_MODEL is empty. Set it in the local .env file.")

    client = Groq(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
    except APIError as exc:
        raise GroqRequestError(f"Groq API request failed: {exc}") from exc

    answer = completion.choices[0].message.content
    if not answer or not answer.strip():
        raise GroqRequestError("Groq returned an empty answer.")
    return answer.strip()
