import pytest
from test_helpers.tasks import minimal_task_for_tool_use
from test_helpers.tool_call_utils import get_tool_call, get_tool_response
from test_helpers.utils import skip_if_no_docker

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.tool import bash


@skip_if_no_docker
@pytest.mark.slow
def test_bash_simple_echo() -> None:
    task = minimal_task_for_tool_use(bash())
    result = eval(
        task,
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name=bash.__name__,
                    tool_arguments={"cmd": "echo 'testing bash tool'"},
                ),
            ],
        ),
    )[0]
    assert result.samples
    messages = result.samples[0].messages
    tool_call = get_tool_call(messages, bash.__name__)
    assert tool_call is not None
    tool_call_response = get_tool_response(messages, tool_call)
    assert tool_call_response is not None
    assert tool_call_response.content == "testing bash tool\n"
