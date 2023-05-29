import argparse
import inspect
import typing
from typing import Callable, ClassVar, Dict, Iterator, Optional, Sequence, Type, Union


class Workflow:
    _registry: ClassVar[Dict[str, Type["Workflow"]]] = {}

    @classmethod
    def register(self, name: str, exist_ok: bool = False) -> Callable[[Type["Workflow"]], Type["Workflow"]]:
        def wrapper(workflow: Type["Workflow"]) -> Type["Workflow"]:
            if not exist_ok and name in self._registry:
                raise ValueError(f"Workflow '{name}' was already registered.")

            self._registry[name] = workflow
            return workflow

        return wrapper

    @classmethod
    def by_name(cls, name: str) -> Type["Workflow"]:
        return cls._registry[name]

    @classmethod
    def available_names(cls) -> Sequence[str]:
        return list(cls._registry)

    @staticmethod
    def _setup_parser(parser: argparse.ArgumentParser, func: Callable) -> argparse.ArgumentParser:
        parser.set_defaults(__func=func)
        parser.description = func.__doc__

        signature = inspect.signature(func)

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str

            optional = param.default != inspect.Parameter.empty
            default = param.default if optional else None

            origin = typing.get_origin(arg_type)
            args = typing.get_args(arg_type)
            if origin == Union and len(args) == 2 and args[1] == type(None):  # noqa: E721
                arg_type = args[0]
                optional = True
                default = None

            positional = param.kind in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
            )

            help_message = f"{arg_type.__name__}" if arg_type else "str"
            if optional:
                help_message += f" (default: {default})"
            elif not positional:
                help_message += " (required)"

            argparse_kwargs = {
                "help": help_message,
                "type": arg_type,
            }
            if optional:
                argparse_kwargs["default"] = default
            elif not positional:
                argparse_kwargs["required"] = True

            if positional:
                parser.add_argument(name, **argparse_kwargs)
            else:
                name = name.replace("_", "-")
                parser.add_argument("--" + name, **argparse_kwargs)

        return parser

    @classmethod
    def _collect_methods(cls) -> Iterator[Callable]:
        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_") and not inspect.isclass(func):
                yield func

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        subparsers = parser.add_subparsers()

        for func in cls._collect_methods():
            subparser = subparsers.add_parser(func.__name__, help=func.__doc__)
            cls._setup_parser(subparser, func)

        return parser

    @classmethod
    def run(cls, args: Optional[Sequence[str]] = None) -> None:
        args = args or ["--help"]
        parser = cls.build_parser()
        namespace = parser.parse_args(args)
        params = vars(namespace)
        func = params.pop("__func")
        kwargs = {k.replace("-", "_"): v for k, v in params.items()}
        func(cls(), **kwargs)
