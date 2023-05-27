import argparse
import inspect
import typing
from typing import Callable, Iterator, Optional, Sequence, Union


class Workflow:
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

            argparse_kwargs = {
                "help": help_message,
                "type": arg_type,
                "default": default,
            }

            if positional:
                parser.add_argument(name, **argparse_kwargs)
            else:
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
        func(cls(), **params)


class MyWorkflow(Workflow):
    """My workflow."""

    def greet(
        self,
        name: str,
        *,
        greeting: str = "Hello",
    ) -> None:
        """Greet someone."""
        print(f"{greeting}, {name}!")


if __name__ == "__main__":
    import importlib

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("workflow", nargs="?", help="workflow name to run formatted as module:classname")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments to pass to the workflow")
    parser.add_argument("--help", action="store_true", help="show this help message and exit")
    args = parser.parse_args()

    if not args.workflow:
        parser.print_help()
        exit(1)

    workflowpath = args.workflow
    if ":" not in workflowpath:
        workflowpath = f"__main__:{workflowpath}"

    modulename, classname = workflowpath.rsplit(":", 1)
    try:
        module = importlib.import_module(modulename)
        workflow = getattr(module, classname)
        if workflow is None:
            raise AttributeError
    except (ImportError, AttributeError):
        print(f"Could not find workflow {workflowpath}")
        exit(1)

    if inspect.getmro(workflow)[1] is Workflow:
        print(f"{args.workflow} is not a subclass of Workflow")
        exit(1)

    if args.help:
        args.args = ["--help"]

    workflow.run(args.args)
