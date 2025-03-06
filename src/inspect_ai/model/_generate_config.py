from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from inspect_ai.util._json import JSONSchema


class ResponseSchema(BaseModel):
    """Schema for model response when using Structured Output."""

    name: str
    """The name of the response schema. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64."""

    json_schema: JSONSchema
    """The schema for the response format, described as a JSON Schema object."""

    description: str | None = Field(default=None)
    """A description of what the response format is for, used by the model to determine how to respond in the format."""

    strict: bool | None = Field(default=None)
    """Whether to enable strict schema adherence when generating the output. If set to true, the model will always follow the exact schema defined in the schema field.
    OpenAI and Mistral only."""


class GenerateConfigArgs(TypedDict, total=False):
    """Type for kwargs that selectively override GenerateConfig."""

    max_retries: int | None
    """Maximum number of times to retry request (defaults to 5)."""

    timeout: int | None
    """Request timeout (in seconds)."""

    max_connections: int | None
    """Maximum number of concurrent connections to Model API (default is model specific)."""

    system_message: str | None
    """Override the default system message."""

    max_tokens: int | None
    """The maximum number of tokens that can be generated in the completion (default is model specific)."""

    top_p: float | None
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass."""

    temperature: float | None
    """What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."""

    stop_seqs: list[str] | None
    """Sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence."""

    best_of: int | None
    """Generates best_of completions server-side and returns the 'best' (the one with the highest log probability per token). vLLM only."""

    frequency_penalty: float | None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. OpenAI, Google, Grok, Groq, and vLLM only."""

    presence_penalty: float | None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. OpenAI, Google, Grok, Groq, and vLLM only."""

    logit_bias: dict[int, float] | None
    """Map token Ids to an associated bias value from -100 to 100 (e.g. "42=10,43=-10"). OpenAI and Grok only."""

    seed: int | None
    """Random seed. OpenAI, Google, Mistral, Groq, HuggingFace, and vLLM only."""

    top_k: int | None
    """Randomly sample the next word from the top_k most likely next words. Anthropic, Google, and HuggingFace only."""

    num_choices: int | None
    """How many chat completion choices to generate for each input message. OpenAI, Grok, Google, and TogetherAI only."""

    logprobs: bool | None
    """Return log probabilities of the output tokens. OpenAI, Grok, TogetherAI, Huggingface, llama-cpp-python, and vLLM only."""

    top_logprobs: int | None
    """Number of most likely tokens (0-20) to return at each token position, each with an associated log probability. OpenAI, Grok, and Huggingface only."""

    parallel_tool_calls: bool | None
    """Whether to enable parallel function calling during tool use (defaults to True). OpenAI and Groq only."""

    internal_tools: bool | None
    """Whether to automatically map tools to model internal implementations (e.g. 'computer' for anthropic)."""

    max_tool_output: int | None
    """Maximum tool output (in bytes). Defaults to 16 * 1024."""

    cache_prompt: Literal["auto"] | bool | None
    """Whether to cache the prompt prefix. Defaults to "auto", which will enable caching for requests with tools. Anthropic only."""

    reasoning_effort: Literal["low", "medium", "high"] | None
    """Constrains effort on reasoning for reasoning models. Open AI o1 models only."""

    reasoning_tokens: int | None
    """Maximum number of tokens to use for reasoning. Anthropic Claude models only."""

    reasoning_history: Literal["none", "all", "last", "auto"] | None
    """Include reasoning in chat message history sent to generate."""

    response_schema: ResponseSchema | None
    """Request a response format as JSONSchema (output should still be validated). OpenAI, Google, and Mistral only."""


class GenerateConfig(BaseModel):
    """Model generation options."""

    max_retries: int | None = Field(default=None)
    """Maximum number of times to retry request (defaults to 5)."""

    timeout: int | None = Field(default=None)
    """Request timeout (in seconds)."""

    max_connections: int | None = Field(default=None)
    """Maximum number of concurrent connections to Model API (default is model specific)."""

    system_message: str | None = Field(default=None)
    """Override the default system message."""

    max_tokens: int | None = Field(default=None)
    """The maximum number of tokens that can be generated in the completion (default is model specific)."""

    top_p: float | None = Field(default=None)
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass."""

    temperature: float | None = Field(default=None)
    """What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."""

    stop_seqs: list[str] | None = Field(default=None)
    """Sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence."""

    best_of: int | None = Field(default=None)
    """Generates best_of completions server-side and returns the 'best' (the one with the highest log probability per token). vLLM only."""

    frequency_penalty: float | None = Field(default=None)
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. OpenAI, Google, Grok, Groq, and vLLM only."""

    presence_penalty: float | None = Field(default=None)
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. OpenAI, Google, Grok, Groq, and vLLM only."""

    logit_bias: dict[int, float] | None = Field(default=None)
    """Map token Ids to an associated bias value from -100 to 100 (e.g. "42=10,43=-10"). OpenAI, Grok, and Grok only."""

    seed: int | None = Field(default=None)
    """Random seed. OpenAI, Google, Mistral, Groq, HuggingFace, and vLLM only."""

    top_k: int | None = Field(default=None)
    """Randomly sample the next word from the top_k most likely next words. Anthropic, Google, HuggingFace, and vLLM only."""

    num_choices: int | None = Field(default=None)
    """How many chat completion choices to generate for each input message. OpenAI, Grok, Google, TogetherAI, and vLLM only."""

    logprobs: bool | None = Field(default=None)
    """Return log probabilities of the output tokens. OpenAI, Grok, TogetherAI, Huggingface, llama-cpp-python, and vLLM only."""

    top_logprobs: int | None = Field(default=None)
    """Number of most likely tokens (0-20) to return at each token position, each with an associated log probability. OpenAI, Grok, Huggingface, and vLLM only."""

    parallel_tool_calls: bool | None = Field(default=None)
    """Whether to enable parallel function calling during tool use (defaults to True). OpenAI and Groq only."""

    internal_tools: bool | None = Field(default=None)
    """Whether to automatically map tools to model internal implementations (e.g. 'computer' for anthropic)."""

    max_tool_output: int | None = Field(default=None)
    """Maximum tool output (in bytes). Defaults to 16 * 1024."""

    cache_prompt: Literal["auto"] | bool | None = Field(default=None)
    """Whether to cache the prompt prefix. Defaults to "auto", which will enable caching for requests with tools. Anthropic only."""

    reasoning_effort: Literal["low", "medium", "high"] | None = Field(default=None)
    """Constrains effort on reasoning for reasoning models. Open AI o1 models only."""

    reasoning_tokens: int | None = Field(default=None)
    """Maximum number of tokens to use for reasoning. Anthropic Claude models only."""

    reasoning_history: Literal["none", "all", "last", "auto"] | None = Field(
        default=None
    )
    """Include reasoning in chat message history sent to generate."""

    response_schema: ResponseSchema | None = Field(default=None)
    """Request a response format as JSONSchema (output should still be validated). OpenAI, Google, and Mistral only."""

    # migrate reasoning_history as a bool
    @model_validator(mode="before")
    @classmethod
    def migrate_reasoning(cls, data: Any) -> Any:
        if isinstance(data, dict):
            reasoning_history = data.get("reasoning_history", None)
            if reasoning_history is True:
                data["reasoning_history"] = "all"
            elif reasoning_history is False:
                data["reasoning_history"] = "none"

        return data

    def merge(
        self, other: Union["GenerateConfig", GenerateConfigArgs]
    ) -> "GenerateConfig":
        """Merge another model configuration into this one.

        Args:
           other (Union[GenerateConfig, GenerateConfigArgs]):
              Configuration to merge.

        Returns:
           Merged configuration.
        """
        if not isinstance(other, GenerateConfig):
            other = GenerateConfig(**other)
        config_keys = list(GenerateConfigArgs.__mutable_keys__)  # type: ignore
        config = deepcopy(self)
        for key in config_keys:
            value = getattr(other, key, None)
            if value is not None:
                setattr(config, key, value)
        return config


def active_generate_config() -> GenerateConfig:
    return active_generate_config_context_var.get()


def set_active_generate_config(config: GenerateConfig) -> None:
    active_generate_config_context_var.set(config)


active_generate_config_context_var: ContextVar[GenerateConfig] = ContextVar(
    "generate_config", default=GenerateConfig()
)
