from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstract base class for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt.
        Must include any required stop-token logic at the caller level.
        """
        raise NotImplementedError


class OpenAIModel(LLM):
    """
    Example LLM implementation using OpenAI's Responses API.

    TODO(student): Implement this class to call your chosen backend (e.g., OpenAI GPT-5 mini)
    and return the model's text output. You should ensure the model produces the response
    format required by ResponseParser and include the stop token in the output string.
    """

    def __init__(self, stop_token: str, model_name: str = "gpt-5-mini"):
        # Initialize OpenAI client; API key is read from environment
        from openai import OpenAI
        self.stop_token = stop_token
        self.model_name = model_name
        self._client = OpenAI()

    def generate(self, prompt: str) -> str:
        # Call the model, obtain text, and ensure the stop token is present at the end
        text: str = ""
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            # Be defensive: choices may be missing/empty
            if getattr(completion, "choices", None) and len(completion.choices) > 0:
                choice0 = completion.choices[0]
                msg = getattr(choice0, "message", None)
                if msg and getattr(msg, "content", None):
                    text = msg.content
                elif getattr(choice0, "text", None):
                    text = choice0.text  # some providers use 'text'
                else:
                    text = ""
            else:
                text = ""
        except Exception:
            # Fallback to responses API if chat.completions is unavailable
            try:
                response = self._client.responses.create(
                    model=self.model_name,
                    tools=[{ "type": "web_search_preview" }],
                    input=prompt,
                )
                # Try multiple extraction strategies depending on SDK version
                extracted = getattr(response, "output_text", None)
                if not extracted:
                    # Attempt to read from response.output list
                    output = getattr(response, "output", None)
                    if output and len(output) > 0:
                        first = output[0]
                        content = getattr(first, "content", None)
                        if content and len(content) > 0:
                            maybe_text = getattr(content[0], "text", None)
                            extracted = getattr(maybe_text, "value", None) or getattr(maybe_text, "content", None)
                text = extracted or ""
            except Exception as e:
                raise e

        if text is None:
            text = ""
        # Ensure required stop token is present so the parser can find END_CALL
        stripped = text.rstrip()
        if not stripped.endswith(self.stop_token):
            if stripped:
                text = stripped + "\n" + self.stop_token
            else:
                text = self.stop_token
        return text