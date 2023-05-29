from __future__ import annotations

import functools
import os
import re
import sys
import time
from collections.abc import Sized
from typing import Any, Callable, ClassVar, Generic, Iterable, Iterator, TextIO, TypeVar, cast

T = TypeVar("T")
Self = TypeVar("Self", bound="ProgressBar")

DISABLE_PROGRESSBAR = os.environ.get("NLPSTACK_DISABLE_PROGRESSBAR", "0").lower() in (
    "1",
    "true",
)
ANSI_COLOR = re.compile(r"(\033|\x1b)\[[0-9;]*m")


def _dummy_iterator() -> Iterator[int]:
    iterations = 0
    while True:
        yield iterations
        iterations += 1


def _default_sizeof_formatter(size: int | float) -> str:
    if isinstance(size, int):
        return str(size)
    return f"{size:.1f}"


def _truncate_text(text: str, width: int) -> str:
    if len(re.sub(ANSI_COLOR, "", text)) <= width:
        return text

    position = 0
    visible_length = 0
    truncated_text = ""
    for match in re.finditer(ANSI_COLOR, text):
        steps = min(match.start() - position, width - visible_length - 1)
        truncated_text += text[position : position + steps]
        truncated_text += text[match.start() : match.end()]
        visible_length += steps
        position = match.end()

    if visible_length < width:
        truncated_text += text[position : position + width - visible_length - 1]

    return truncated_text + "…"


class EMA:
    def __init__(
        self,
        alpha: float = 0.3,
    ) -> None:
        self._alpha = alpha
        self._value = 0.0

    def update(self, value: float) -> None:
        self._value = self._alpha * value + (1.0 - self._alpha) * self._value

    def get(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


class ProgressBar(Generic[T]):
    _session: ClassVar[list[ProgressBar[Any]]] = []

    def __init__(
        self,
        total_or_iterable: int | Iterable[T] | None,
        desc: str | None = None,
        unit: str = "it",
        leave: bool = True,
        position: int | None = None,
        output: TextIO = sys.stderr,
        maxwidth: int | None = None,
        truncate: bool = True,
        framerate: float = 16.0,
        template: str | None = None,
        maxbarwidth: int | None = 40,
        sizeof_formatter: Callable[[int | float], str] = _default_sizeof_formatter,
        disable: bool = False,
    ) -> None:
        total_or_iterable = total_or_iterable or cast(Iterator[T], _dummy_iterator())
        position = position if position is not None else len(self.get_active_progressbars())
        self._iterable = (
            cast(Iterator[T], range(total_or_iterable)) if isinstance(total_or_iterable, int) else total_or_iterable
        )
        self._total = len(self._iterable) if isinstance(self._iterable, Sized) else None
        self._desc = desc
        self._unit = unit
        self._leave = leave
        self._position = position
        self._output = output
        self._maxwidth = maxwidth
        self._truncate = truncate
        self._framerate = framerate
        self._template = template
        self._maxbarwidth = maxbarwidth
        self._sizeof_formatter = sizeof_formatter
        self._disable = disable or DISABLE_PROGRESSBAR

        self._postfixes: dict[str, Any] = {}

        self._iterations = 0
        self._start_time = time.time()
        self._last_time = self._start_time
        self._last_time_rendered = self._start_time
        self._interval_ema = EMA()

        self._write = getattr(output, "__original_write__", output.write)
        self._flush = getattr(output, "__original_flush__", output.flush)

        self._is_already_entered = False
        self._is_already_exited = False
        self._is_prevented_from_exiting = False

    def is_active(self) -> bool:
        return self._is_already_entered and not self._is_already_exited

    @classmethod
    def get_session(cls) -> list[ProgressBar[Any]]:
        return cls._session

    @classmethod
    def get_active_progressbars(cls) -> list[ProgressBar[Any]]:
        return [pb for pb in cls._session if pb.is_active()]

    @classmethod
    def get_enabled_progressbars(cls) -> list[ProgressBar[Any]]:
        return [pb for pb in cls._session if not pb._disable]

    @classmethod
    def get_active_max_position(cls) -> int:
        active_progressbars = cls.get_active_progressbars()
        if not active_progressbars:
            return 0
        return max(pb._position for pb in active_progressbars)

    @classmethod
    def get_session_max_position(cls) -> int:
        enabled_progressbars = cls.get_enabled_progressbars()
        if not enabled_progressbars:
            return 0
        return max(pb._position for pb in enabled_progressbars)

    @staticmethod
    def _format_time(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{int(h):d}:{int(m):02d}:{int(s):02d}"
        return f"{int(m):02d}:{int(s):02d}"

    def _get_maxwidth(self) -> int:
        try:
            terminal_width, _ = os.get_terminal_size()
        except OSError:
            terminal_width = 80
        if self._maxwidth:
            return min(terminal_width, self._maxwidth)
        return terminal_width

    def _get_bar(self, width: int, percentage: float) -> str:
        width = max(1, width)
        if self._maxbarwidth is not None:
            width = min(width, self._maxbarwidth)
        ratio = percentage / 100
        current_width = int(ratio * width)
        remaining_width = width - current_width
        if ratio < 1.0 and remaining_width > 0:
            bar = f"\033[35m{'━' * current_width}╸\033[0m\033[30m{'━' * (remaining_width - 1)}\033[0m"
        elif remaining_width == 0:
            bar = f"\033[32m{'━' * width}\033[0m"
        else:
            bar = f"\033[31m!{'━' * (width - 1)}\033[0m"
        return bar

    def set_description(self, desc: str | None = None) -> None:
        self._desc = desc

    def set_postfix(self, **postfixes: Any) -> None:
        self._postfixes = postfixes

    def show(self, *, force: bool = True) -> None:
        if self._disable:
            return
        if not self._leave and self._is_already_exited:
            return

        current_time = time.time() if self.is_active() else self._last_time

        if not force and self._iterations > 0 and self._iterations < (self._total or float("inf")):
            framerate = 1.0 / (current_time - self._last_time_rendered + 1.0e-13)
            if framerate > self._framerate:
                return

        do_compose_template = not self._template

        template = self._template or ""
        contents: dict[str, Any] = {}

        elapsed_time = current_time - self._start_time
        interval_ema = self._interval_ema.get()
        average_iterations = 1.0 / interval_ema if interval_ema > 0.0 else 0.0

        contents["desc"] = self._desc
        contents["unit"] = self._unit
        contents["iterations"] = self._sizeof_formatter(self._iterations)
        contents["elapsed_time"] = self._format_time(elapsed_time)
        contents["average_iterations"] = self._sizeof_formatter(average_iterations)

        postfixes = [f"{key}={val}" for key, val in self._postfixes.items()]

        if self._desc:
            contents["desc"] = self._desc
            if do_compose_template:
                template = "{desc}  " + template

        if self._total is None:
            postfix_template = " ".join(postfixes)
            if do_compose_template:
                template = template + " {iterations}{unit}  {elapsed_time}"
            if postfix_template:
                template += "  " + postfix_template
        else:
            total_width = len(self._sizeof_formatter(self._total))
            percentage = int(100 * self._iterations / self._total)
            remaining_time = (self._total - self._iterations) * interval_ema
            postfix_template = " ".join(postfixes)

            if percentage > 100:
                remaining_time = 0.0

            if do_compose_template:
                template = template + "{percentage:3d}%  {bar}  {elapsed_time}<{remaining_time}"
            if postfix_template:
                template += "  " + postfix_template

            contents["total_width"] = total_width
            contents["percentage"] = percentage
            contents["bar"] = ""
            contents["total"] = self._sizeof_formatter(self._total)
            contents["remaining_time"] = self._format_time(remaining_time)

            barwidth = max(3, self._get_maxwidth() - len(template.format(**contents)))
            contents["bar"] = self._get_bar(barwidth, percentage)

        line = template.format(**contents)
        if self._truncate:
            line = _truncate_text(line, self._get_maxwidth())

        self._write("\033[B" * self._position)
        self._write(f"\x1b[2K\r{line}")
        self._write("\033[A\r" * self._position)
        self._flush()

        self._last_time_rendered = current_time

    def update(self, iterations: int = 1) -> None:
        current_time = time.time()
        self._iterations += iterations
        self._last_time = current_time
        self._interval_ema.update((current_time - self._start_time) / self._iterations)
        for progress in ProgressBar.get_active_progressbars():
            progress.show()

    def __iter__(self) -> Iterator[T]:
        self._iterations = 0
        self._start_time = time.time()

        try:
            if self._is_already_entered:
                self._is_prevented_from_exiting = True
            with self:
                for item in self._iterable:
                    yield item
                    self.update()
        finally:
            self._is_prevented_from_exiting = False

    def __enter__(self: Self) -> Self:
        def write_wrapper(file: TextIO, text: str) -> int:
            newline_count = text.count("\n")
            active_max_position = ProgressBar.get_session_max_position()
            self._write("\x1b[2K\r")
            self._write("\x1b[2K\r\033[B" * active_max_position)
            self._write("\n\r\x1b[2K" * newline_count)
            self._write("\033[A" * (active_max_position + newline_count))
            self._write("\r")
            result = int(getattr(file, "__original_write__")(text))
            for pb in ProgressBar.get_session():
                pb.show(force=True)
            return result

        def flush_wrapper(file: TextIO) -> None:
            self._flush()
            if file is not self._output:
                getattr(file, "__original_flush__")()

        if not self._is_already_entered:
            self._iterations = 0
            self._start_time = time.time()
            if not self._disable and self._output in (sys.stdout, sys.stderr):
                for file in {sys.stderr, sys.stdout}:
                    if not hasattr(file, "__original_write__"):
                        setattr(file, "__original_write__", file.write)
                        setattr(file, "write", functools.partial(write_wrapper, file))
                    if not hasattr(file, "__original_flush__"):
                        setattr(file, "__original_flush__", file.flush)
                        setattr(file, "flush", functools.partial(flush_wrapper, file))

                self._write("\x1b[?25l")
                self._write("\n" * self._position)
                self._write("\x1b[A" * self._position)
                self.show()
            ProgressBar._session.append(self)
            self._is_already_entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._is_prevented_from_exiting:
            return
        if not self._is_already_exited:
            active_progresses = ProgressBar.get_active_progressbars()
            active_progresses.remove(self)
            if not self._leave:
                ProgressBar._session.remove(self)
            if not self._disable:
                self.show()
                if not self._leave:
                    self._write("\033[B" * self._position)
                    self._write("\x1b[2K\r")
                    self._write("\033[A" * self._position)
                if not active_progresses:
                    self._write("\033[B" * (ProgressBar.get_session_max_position() - self._position))
                    self._write("\n\x1b[?25h")
                self._flush()

                if not active_progresses and self._output in (
                    sys.stdout,
                    sys.stderr,
                ):
                    for file in {sys.stderr, sys.stdout}:
                        if hasattr(file, "__original_write__"):
                            setattr(file, "write", file.__original_write__)
                            delattr(file, "__original_write__")
                        if hasattr(file, "__original_flush__"):
                            setattr(file, "flush", file.__original_flush__)
                            delattr(file, "__original_flush__")

                    ProgressBar._session.clear()
            self._is_already_exited = True
