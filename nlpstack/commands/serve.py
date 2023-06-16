import argparse
import functools
import pickle
from http.server import HTTPServer
from logging import getLogger

import minato

from nlpstack.rune import Rune
from nlpstack.server.handler import RuneHandler

from .subcommand import Subcommand

logger = getLogger(__name__)


@Subcommand.register("serve")
class ServeCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument(
            "archive_filename",
            type=str,
            help="Path to rune archive file",
        )
        self.parser.add_argument(
            "--port",
            type=int,
            default=8080,
            help="Port to listen on (default: %(default)s)",
        )
        self.parser.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="Host to listen on (default: %(default)s)",
        )

    def run(self, args: argparse.Namespace) -> None:
        with minato.open(args.archive_filename, "rb") as fp:
            rune = pickle.load(fp)

        if not isinstance(rune, Rune):
            print("Given archive file is not a rune archive")
            exit(1)

        server = HTTPServer(
            (args.host, args.port),
            functools.partial(RuneHandler, rune=rune),
        )
        logger.info("Listening on %s:%d", args.host, args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            server.shutdown()
            logger.info("Done")
