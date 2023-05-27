import argparse
import importlib

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

        if not issubclass(workflow, Workflow):
            print(f"{args.workflow} is not a subclass of Workflow")
            exit(1)

        workflow.run(args.args)
