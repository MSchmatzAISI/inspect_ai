import inspect
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    cast,
    overload,
    runtime_checkable,
)

from typing_extensions import Unpack

from inspect_ai._util._async import is_callable_coroutine
from inspect_ai._util.interrupt import check_sample_interrupt
from inspect_ai._util.registry import (
    RegistryInfo,
    registry_add,
    registry_create,
    registry_name,
    registry_tag,
)
from inspect_ai.agent._agent import Agent, is_agent
from inspect_ai.agent._as_solver import as_solver
from inspect_ai.model import CachePolicy, GenerateConfigArgs

from ._task_state import TaskState, set_sample_state


@runtime_checkable
class Generate(Protocol):
    async def __call__(
        self,
        state: TaskState,
        tool_calls: Literal["loop", "single", "none"] = "loop",
        cache: bool | CachePolicy = False,
        **kwargs: Unpack[GenerateConfigArgs],
    ) -> TaskState:
        """Generate using the model and add the assistant message to the task state.

        Args:
            state: Beginning task state.
            tool_calls:
                - `"loop"` resolves tools calls and then invokes `generate()`,
                    proceeding in a loop which terminates when there are no more
                    tool calls, or `message_limit` or `token_limit` is exceeded.
                    This is the default behavior.
                - `"single"` resolves at most a single set of tool calls and then returns.
                - `"none"` does not resolve tool calls at all (in this
                    case you will need to invoke `call_tools()` directly).
            cache: Caching behaviour for generate responses (defaults to no caching).
            **kwargs: Optional generation config arguments.

        Returns:
            Updated TaskState.
        """
        ...


@dataclass(frozen=True)
class SolverSpec:
    """Solver specification used to (re-)create solvers."""

    solver: str
    """Solver name (simple name or file.py@name)."""

    args: dict[str, Any] = field(default_factory=dict)
    """Solver arguments."""


@runtime_checkable
class Solver(Protocol):
    async def __call__(
        self,
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        r"""Contribute to solving an evaluation task.

        Transform a `TaskState`, returning the new state. Solvers may
        optionally call the `generate()` function to create a new
        state resulting from model generation. Solvers may also do
        prompt engineering or other types of elicitation.

        Args:
          state: State for tasks being evaluated.
          generate: Function for generating outputs.

        Returns:
          Updated TaskState.

        Examples:
          ```python
          @solver
          def prompt_cot(template: str) -> Solver:
              def solve(state: TaskState, generate: Generate) -> TaskState:
                  # insert chain of thought prompt
                  return state

              return solve
          ```
        """
        ...


P = ParamSpec("P")


def solver_register(solver: Callable[P, Solver], name: str = "") -> Callable[P, Solver]:
    r"""Register a function or class as a solver.

    Args:
        solver (Callable[P, Solver]):
            Function that returns a Solver or class derived Solver.
        name (str): Name of solver (Optional, defaults to object name)

    Returns:
        Solver with registry attributes.
    """
    solver_name = name if name else getattr(solver, "__name__")
    registry_add(solver, RegistryInfo(type="solver", name=solver_name))
    return solver


def solver_create(name: str, **kwargs: Any) -> Solver:
    r"""Create a Solver based on its registered name.

    Args:
        name (str): Name of solver (Optional, defaults to object name)
        **kwargs (dict): Optional creation arguments for the solver

    Returns:
        Solver with registry info attribute
    """
    return registry_create("solver", name, **kwargs)


SolverType: TypeAlias = Solver | Agent
"""Return type for @solver decorated functions. """


@overload
def solver(name: str) -> Callable[[Callable[P, Solver]], Callable[P, Solver]]: ...


@overload
def solver(name: Callable[P, SolverType]) -> Callable[P, Solver]: ...


def solver(
    name: str | Callable[P, SolverType],
) -> Callable[[Callable[P, Solver]], Callable[P, Solver]] | Callable[P, Solver]:
    r"""Decorator for registering solvers.

    Args:
        name:
            Optional name for solver. If the decorator has no name
            argument then the name of the underlying Callable[P, SolverType]
            object will be used to automatically assign a name.

    Returns:
        Solver with registry attributes.

    Examples:
        ```python
        @solver
        def prompt_cot(template: str) -> Solver:
            def solve(state: TaskState, generate: Generate) -> TaskState:
                # insert chain of thought prompt
                return state

            return solve
        ```
    """

    # create_solver_wrapper:
    #  (a) Add the SolverType to the registry using the appropriately
    #      package-namespaced name
    #  (b) Ensure that instances of Solver created by SolverType also
    #      carry registry info.
    def create_solver_wrapper(
        solver_type: Callable[P, SolverType], name: str | None = None
    ) -> Callable[P, Solver]:
        solver_name = registry_name(
            solver_type, name if name else getattr(solver_type, "__name__")
        )

        @wraps(solver_type)
        def solver_wrapper(*args: P.args, **kwargs: P.kwargs) -> Solver:
            solver = solver_type(*args, **kwargs)
            if is_agent(solver):
                solver = as_solver(solver)
            solver = cast(Solver, solver)

            if not is_callable_coroutine(solver):
                raise TypeError(f"'{solver}' is not declared as an async callable.")

            # if the solver is a class then we inject state tracking
            # by patching the __call__ method (this is because we
            # want to preserve the type, especially for code that e.g.
            # checks for Chain or Plan)
            if inspect.isclass(type(solver)):
                original_call = solver.__call__

                @wraps(original_call)
                async def call_with_state(
                    state: TaskState, generate: Generate
                ) -> TaskState:
                    state = await original_call(state, generate)
                    check_sample_interrupt()
                    set_sample_state(state)
                    return state

                registered_solver = solver
                setattr(registered_solver, "__call__", call_with_state)

            # if its a function then use ordinary @wraps to preserve
            # the wrapped solver
            else:

                @wraps(solver)
                async def registered_solver(
                    state: TaskState, generate: Generate
                ) -> TaskState:
                    state = await solver(state, generate)
                    check_sample_interrupt()
                    set_sample_state(state)
                    return state

            registry_tag(
                solver_type,
                registered_solver,
                RegistryInfo(type="solver", name=solver_name),
                *args,
                **kwargs,
            )

            return registered_solver

        # functools.wraps overrides the return type annotation of the inner function, so
        # we explicitly set it again
        solver_wrapper.__annotations__["return"] = Solver

        return solver_register(cast(Callable[P, Solver], solver_wrapper), solver_name)

    # for decorators with an explicit name, one more wrapper for the name
    if isinstance(name, str):

        def wrapper(solver_type: Callable[..., Solver]) -> Callable[..., Solver]:
            return create_solver_wrapper(solver_type, name)

        return wrapper

    # create a solver wrapper for the passed solver_type
    else:
        solver_type = name
        return create_solver_wrapper(solver_type)


@solver
def generate(
    tool_calls: Literal["loop", "single", "none"] = "loop",
    cache: bool | CachePolicy = False,
    **kwargs: Unpack[GenerateConfigArgs],
) -> Solver:
    r"""Generate output from the model and append it to task message history.

    generate() is the default solver if none is specified for a given task.

    Args:
      tool_calls (Literal["loop", "single", "none"]): Resolve tool calls:
        - `"loop"` resolves tools calls and then invokes `generate()`,
            proceeding in a loop which terminates when there are no more
            tool calls or `message_limit` or `token_limit` is exceeded.
            This is the default behavior.
        - `"single"` resolves at most a single set of tool calls and then returns.
        - `"none"` does not resolve tool calls at all (in this
            case you will need to invoke `call_tools()` directly).

      cache: (bool | CachePolicy):
        Caching behaviour for generate responses (defaults to no caching).

      **kwargs: Optional generation config arguments.
    """

    # call generate on the tasks
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate(state, tool_calls=tool_calls, cache=cache, **kwargs)

    # return solve
    return solve
