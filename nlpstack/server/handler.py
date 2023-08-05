import dataclasses
import json
import typing
from http.server import SimpleHTTPRequestHandler
from logging import getLogger
from pathlib import Path
from typing import Any, Generic, Tuple, Type, TypeVar

from colt import ColtBuilder
from colt.error import ConfigurationError

from nlpstack.common import generate_json_schema
from nlpstack.rune import Rune

logger = getLogger(__name__)
coltbuilder = ColtBuilder(typekey="type", strict=True)

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")


class RuneHandler(SimpleHTTPRequestHandler, Generic[Example, Prediction]):
    @staticmethod
    def _extract_example_and_prediction_classes(
        rune: Rune[Example, Prediction]
    ) -> Tuple[Type[Example], Type[Prediction]]:
        bases = getattr(rune, "__orig_bases__", None)
        if bases is None:
            raise ValueError("Rune must be a generic type")

        rune_bases = [
            base
            for base in bases
            if (isinstance(base, typing._GenericAlias) and issubclass(base.__origin__, Rune))  # type: ignore
        ]
        if len(bases) < 1:
            raise ValueError("Rune must be a subclass of Rune")

        rune_base = rune_bases[0]
        rune_args = typing.get_args(rune_base)
        if len(rune_args) < 2:
            raise ValueError("Rune must have two type arguments")

        input_class, output_class = rune_args[:2]
        if not dataclasses.is_dataclass(input_class):
            raise ValueError("Input class must be a dataclass")
        if not dataclasses.is_dataclass(output_class):
            raise ValueError("Output class must be a dataclass")

        return input_class, output_class

    def __init__(
        self,
        *args: Any,
        rune: Rune[Example, Prediction],
        health_check: str = "/health",
        **kwargs: Any,
    ) -> None:
        input_class, output_class = self._extract_example_and_prediction_classes(rune)

        self._rune = rune
        self._health_check = health_check
        self._input_class = input_class
        self._output_class = output_class
        self._input_schema = json.dumps(generate_json_schema(input_class))
        self._output_schema = json.dumps(generate_json_schema(output_class))

        super().__init__(*args, **kwargs)

    def _handle_error_response(self, status_code: int, message: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"message": message}).encode())

    def _serve_health_check(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"message": "OK"}).encode())

    def _serve_index(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        index_filename = Path(__file__).parent / "frontend" / "index.html"
        self.wfile.write(index_filename.read_bytes())

    def _serve_schema(self, schema: str) -> None:
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(schema.encode())

    def _serve_prediction(self, example: Example, **kwargs: Any) -> None:
        prediction = next(self._rune.predict([example], **kwargs))
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                dataclasses.asdict(prediction),  # type: ignore
                ensure_ascii=False,
            ).encode()
        )

    def do_GET(self) -> None:
        """Serve a GET request.

        The following paths are supported:
            / -> root
            /schema/input -> input schema
            /schema/output -> output schema
            /<health_check> -> health check
        """

        try:
            if self.path == "/":
                self._serve_index()
            elif self.path == "/schema/input":
                self._serve_schema(self._input_schema)
            elif self.path == "/schema/output":
                self._serve_schema(self._output_schema)
            elif self.path == self._health_check:
                self._serve_health_check()
            else:
                self._handle_error_response(404, "Not Found")
        except Exception as e:
            logger.exception(e)
            self._handle_error_response(500, "Internal Server Error")

    def do_POST(self) -> None:
        """Serve a POST request.

        The following paths are supported:
            <root>/predict -> process input
        """

        try:
            if self.path == "/predict":
                # check content type
                if self.headers["Content-Type"] != "application/json":
                    self._handle_error_response(400, "Bad Request")
                    return
                content_length = int(self.headers["Content-Length"])
                try:
                    body = json.loads(self.rfile.read(content_length))
                    if not isinstance(body, dict):
                        raise ValueError("Body must be a JSON object")
                    inputs = body.pop("inputs")
                    kwargs = body.pop("params", {})
                    example = coltbuilder(inputs, self._input_class)
                except (json.JSONDecodeError, ValueError, ConfigurationError, KeyError):
                    self._handle_error_response(400, "Bad Request")
                    return
                self._serve_prediction(example, **kwargs)
            else:
                self._handle_error_response(404, "Not Found")
        except Exception as e:
            logger.exception(e)
            self._handle_error_response(500, "Internal Server Error")
