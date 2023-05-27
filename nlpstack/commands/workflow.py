import argparse
import importlib
from typing import Any

from nlpstack.workflows import Workflow

from .subcommand import Subcommand


@Subcommand.register("workflow")
class WorkflowCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument(
            "workflow",
            nargs="?",
            help="workflow name to run formatted as module:classname",
        )
        self.parser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="arguments to pass to the workflow",
        )

    def run(self, args: argparse.Namespace) -> None:
        if not args.workflow:
            self.parser.print_help()
            exit(1)

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
                print(f"Could not find workflow {workflowname}")
                exit(1)

        if not issubclass(workflow, Workflow):
            print(f"{args.workflow} is not a subclass of Workflow")
            exit(1)

        workflow.run(args.args)
