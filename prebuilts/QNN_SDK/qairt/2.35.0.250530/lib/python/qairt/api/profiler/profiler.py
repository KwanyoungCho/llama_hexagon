# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import copy
import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, create_model
from pydantic.dataclasses import dataclass

from qairt.api.profiler.report import (
    EventType,
    FunctionProfilingEvent,
    OpTraceGenerator,
    ProfileLogGenerator,
    ProfilingReport,
    ReportGenerator,
)
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.api import ProfilingLevel, ProfilingOption

# global ACTIVE PROFILER
_ACTIVE_PROFILER = None
g_PROFILER_ARGS = ["profiling_level", "profiling_option"]
_profile_logger = get_logger("qairt.profile")


@dataclass
class ProfilerContext:
    """
    Holds metadata or settings for how profiling should behave, e.g. profiling level, options,
    duration etc. Should be freely extended with more fields as needed.
    """

    level: Optional[ProfilingLevel | str] = None
    """Profiling level. Possible levels are: "basic", "detailed", "client" and "backend". """

    option: Optional[ProfilingOption | str] = None
    """ Profiling Option. Possible options are: "optrace" """

    def clear(self):
        self.level = None
        self.option = None


class Profiler:
    """
    Captures function profiling events for the duration of a session. It can be used as a context manager
    and becomes the active profiler in a "with" block.

    It maintains a profiler context and uses a report generator to create profiling reports.
    """

    def __init__(
        self,
        context: Optional[Dict[str, Any]] = None,
        report_generator: Optional[ReportGenerator] = None,
        event_name_prefix: str = "",
    ):
        """
        Initializes the profiler with a context and a report generator.

        Arguments:
            context: A profiler context which describes how profiling should behave. If no context
                     is provided, then a context is created with a "basic" profiling level.
            report_generator: A report generator which can be used to create profiling reports.
            event_name_prefix: A string to prepend to each profiling event name
        """
        self._context = ProfilerContext(**context) if context else ProfilerContext(level="basic")

        if report_generator is None:
            if self._context.option == ProfilingOption.OPTRACE:
                report_generator = OpTraceGenerator()
            else:
                report_generator = ProfileLogGenerator()

        if not isinstance(report_generator, ReportGenerator):
            raise TypeError("report_generator must be an instance of ReportGenerator")

        self._report_generator = report_generator
        self._events: List[FunctionProfilingEvent] = []
        self._event_counter = -1
        self._previous_active_profiler = None
        self._event_name_prefix = event_name_prefix

    def __enter__(self):
        """Enters a profiler context."""
        global _ACTIVE_PROFILER
        self._previous_active_profiler = _ACTIVE_PROFILER
        _ACTIVE_PROFILER = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits a profiler context."""
        global _ACTIVE_PROFILER
        _ACTIVE_PROFILER = self._previous_active_profiler

        if exc_type:
            _profile_logger.error(f"Exception occurred during profiling: {exc_value}")

    @property
    def context(self):
        """Returns the profiler's current context"""
        return self._context

    def capture_function_event(self, name: str, type: str, data: Any):
        """Captures a function profiling event.

        Args:
            name: The name of the function being profiled.
            type: The type of event (e.g. "start", "end", "exception").
            data: Additional data associated with the event.
        """
        self._event_counter += 1
        name = f"{self._event_name_prefix}.{name}" if self._event_name_prefix else name
        event = FunctionProfilingEvent(
            self._event_counter,
            name,
            EventType(type),
            data,
            level=self._context.level,
            option=self._context.option,
        )
        self._events.append(event)
        _profile_logger.debug(f"Profiler captured event: {event}")

    def get_events(self) -> List[FunctionProfilingEvent]:
        """Returns a list of captured profiling events."""
        return self._events

    def get_event(self, event_id: int = -1):
        """Returns the event with the given id."""
        if event_id > len(self._events):
            raise ValueError(f"Function Event with {event_id} not found")
        return self._events[event_id]

    def generate_report(self, event_id: int = -1):
        """Returns a profiling report based on id"""
        if event_id == -1:
            event_id = self._event_counter  # last event
        event_match = self.get_event(event_id)
        return self._report_generator.generate_report(event_match)

    def generate_reports(self) -> List[ProfilingReport]:
        """Returns a profiling report."""
        return self._report_generator.generate_reports(self._events)  # type: ignore

    def get_reports(self):
        """Returns a generator that yields profiling reports."""
        for event in self._events:
            yield self._report_generator.generate_report(event)


def profile(event_type: str, raw_data_callable: Optional[Callable[..., Any]] = None):
    """
    A decorator factory that returns a decorator for profiling functions.

    Args:
        event_type: The event type to profile (e.g. "timing").
        raw_data_callable: A callable which takes the result of called function and
                           returns any data to be stored in the captured event.

    Returns:
        A decorator that profiles the decorated function.
    """

    def decorator(func):
        """ """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = get_active_profiler()

            if not profiler:
                return func(*args, **kwargs)

            # Check if a function accepts a profiling context arg
            # The arg could either be in the function signature parameters as kwarg
            # or if any param is a pydantic base model, the context must be a model field.
            sig = inspect.signature(func)
            params = sig.parameters

            has_profiler_context = False

            # Check if the function accepts a profiling context arg
            if "profiler_context" in params:
                # If the function accepts a profiling context arg, pass the context to the function
                kwargs["profiler_context"] = profiler.context
                has_profiler_context = True
            else:
                # If the function accepts a pydantic base model, pass the context as a model field
                # TODO: This path will be removed given API signatures will no longer accept pydantic
                # models. Keeping this here until: https://jira-dc.qualcomm.com/jira/browse/AISW-125503
                if not kwargs:  # if there are no keyword arguments
                    # If the function accepts a pydantic base model, pass the context as a model field
                    arg_copy = list(copy.copy(args))
                    for idx, arg in enumerate(args):
                        if isinstance(arg, BaseModel) and getattr(arg, "_accepts_profiling_args", False):
                            new_model = add_profiling_fields(type(arg))
                            instance = new_model(**arg.model_dump())
                            setattr(instance, "profiler_context", profiler.context)
                            arg_copy[idx] = instance
                            has_profiler_context = True
                    args = tuple(arg_copy)
                else:
                    kwarg_copy = kwargs.copy()
                    for kw_name, kwarg in kwarg_copy.items():
                        if isinstance(kwarg, BaseModel) and getattr(kwarg, "_accepts_profiling_args", False):
                            new_model = add_profiling_fields(type(kwarg))
                            instance = new_model(**kwarg.model_dump())
                            setattr(instance, "profiler_context", profiler.context)
                            kwargs[kw_name] = instance
                            has_profiler_context = True

                if not has_profiler_context:
                    # check if the function accepts variadic kwargs
                    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
                        kwargs = add_profiling_fields_dict(profiler.context, **kwargs)
                        has_profiler_context = True
                    else:
                        _profile_logger.debug(
                            f"Profiler context was not passed to the function: {func.__name__}"
                        )

                if has_profiler_context:
                    _profile_logger.debug(f"Profiler context was passed to the function: {func.__name__}")

            # Call the function
            result = func(*args, **kwargs)

            # Capture the profiling event
            if raw_data_callable:
                _profile_logger.debug(f"Using raw data callable: {raw_data_callable.__name__}")
                profiling_data = raw_data_callable(result)
            else:
                profiling_data = getattr(result, "profiling_data", None)

            # only capture if profiler context was passed to the function
            # TODO: Enable default capture even if no context was passed
            if has_profiler_context:
                profiler.capture_function_event(name=func.__name__, type=event_type, data=profiling_data)

            return result

        return wrapper

    return decorator


def get_active_profiler() -> Union[Profiler, None]:
    """Returns a currently active profiler (if used inside a with block), otherwise
    None is returned"""
    return _ACTIVE_PROFILER


def add_profiling_fields_dict(profiler_context: ProfilerContext, **kwargs) -> dict:
    """Add profiling fields level and option to the keyword args"""
    kwargs["profiling_level"] = profiler_context.level if profiler_context else None
    kwargs["profiling_option"] = profiler_context.option if profiler_context else None
    return kwargs


def add_profiling_fields(cls: Type[BaseModel]) -> Type[BaseModel]:
    """Returns a modified pydantic class which support profiling levels and options

    1. A profiler context, profiling level and profiling option is inserted into the class
    2. The underlying class can now call cls.profiling_level or cls.profiling_option properties to retrieve
       the levels and options respectively.
    3. The profiling level and option are relative to the profiling context specified."""

    profile_fields: dict = {
        "profiler_context": (
            Optional[ProfilerContext],
            Field(
                default=None,
                description="Internal field for registering a profiler context. "
                " Used to set levels and options.",
                exclude=True,
            ),
        )
    }

    def profiling_level_func(self) -> ProfilingLevel:
        return self.profiler_context.level if self.profiler_context else None

    def profiling_option_func(self) -> ProfilingOption:
        return self.profiler_context.option if self.profiler_context else None

    # Static new properties
    static_properties = {
        "profiling_level": profiling_level_func,
        "profiling_option": profiling_option_func,
    }

    # Add new fields

    #
    extra_args_mode = cls.model_config["extra"]

    if extra_args_mode != "allow":
        cls.model_config["extra"] = "allow"

    new_model = create_model(cls.__name__, __base__=cls, **profile_fields)

    # Add new properties
    for prop_name, prop_func in static_properties.items():
        if hasattr(new_model, prop_name):
            # do not replace any existing values
            continue
        setattr(new_model, prop_name, property(prop_func))

    _profile_logger.debug(f"Added profiling level and profiling option to class {new_model.__name__}")

    # reset extra mode if changed
    cls.model_config["extra"] = extra_args_mode

    return new_model
