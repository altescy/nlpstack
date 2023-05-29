import argparse
import importlib
from logging import getLogger
from typing import Any

from colt import import_modules

from nlpstack.workflows import Workflow

from .subcommand import Subcommand

logger = getLogger(__name__)


@Subcommand.register("workflow")
class WorkflowCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument(
            "workflow",
            nargs="?",
            help="workflow name to run",
        )
        self.parser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="arguments to pass to the workflow",
        )
        self.parser.add_argument(
            "--include-package",
            action="append",
            help="additional packages to include",
            default=[],
        )

    def run(self, args: argparse.Namespace) -> None:
        if not args.workflow:
            self.parser.print_help()
            exit(1)

        if args.include_package:
            logger.info("Importing packages: %s", args.include_package)
            import_modules(args.include_package)

        workflow: Any

        workflowname = args.workflow
        try:
            workflow = Workflow.by_name(workflowname)
        except KeyError:
            if ":" not in workflowname:
                workflowname = f"__main__:{workflowname}"

            modulename, classname = workflowname.rsplit(":", 1)
            try:
                module = importlib.import_module(modulename)
                workflow = getattr(module, classname)
                if workflow is None:
                    raise AttributeError
            except (ImportError, AttributeError):
                print(f"Could not find workflow: {workflowname}")
                print(f"Please choose from the following: {Workflow.available_names()}")
                print(
                    "If you want to use your own workflow, please set `--include-package`"
                    " option or specify the module path as module:YourWorkflow"
                )
                exit(1)

        if not issubclass(workflow, Workflow):
            print(f"{args.workflow} is not a subclass of Workflow")
            exit(1)

        workflow.run(args.args)
