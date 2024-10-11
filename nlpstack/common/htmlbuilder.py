import base64
import io
from textwrap import dedent
from typing import Literal, Mapping, Optional, Sequence, Union

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except ModuleNotFoundError:
    plt = None


def _update_style(
    attributes: Optional[Mapping[str, str]],
    style: str,
) -> Mapping[str, str]:
    attributes = dict(attributes or {})
    if "style" in attributes:
        if not attributes["style"].endswith(";"):
            attributes["style"] += ";"
        attributes["style"] += " " + style
    else:
        attributes["style"] = style
    return attributes


class BaseHTMLElement:
    def __init__(
        self,
        tag: str,
        content: Union[str, "BaseHTMLElement"] = "",
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.tag = tag
        self.content = content
        self.attributes = attributes or {}

    def __str__(self) -> str:
        attributes = " ".join(f'{key}="{value}"' for key, value in self.attributes.items())
        return f"<{self.tag} {attributes}>{self.content}</{self.tag}>"


class Div(BaseHTMLElement):
    def __init__(
        self,
        *content: Union[str, BaseHTMLElement],
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        content_ = "".join(str(item) for item in content)
        super().__init__("div", content_, attributes)


class Stack(Div):
    def __init__(
        self,
        *content: Union[str, BaseHTMLElement],
        direction: Literal["row", "column"] = "column",
        spacing: int = 0,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        style = f"display: flex; flex-direction: {direction}; gap: {spacing}px;"
        attributes = _update_style(attributes, style)
        super().__init__(*content, attributes=attributes)


class Head(BaseHTMLElement):
    def __init__(
        self,
        content: Union[str, BaseHTMLElement] = "",
        level: Literal[1, 2, 3, 4, 5, 6] = 1,
        attributes: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(f"h{level}", content, attributes)


class Center(Div):
    def __init__(
        self,
        *content: Union[str, BaseHTMLElement],
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        style = "display: flex; justify-content: center; align-items: center;"
        attributes = _update_style(attributes, style)
        super().__init__(*content, attributes=attributes)


class Span(BaseHTMLElement):
    def __init__(
        self,
        content: Union[str, BaseHTMLElement] = "",
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__("span", content, attributes)


class Text(BaseHTMLElement):
    def __init__(
        self,
        content: str,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        content = content.replace("<", "&lt;").replace(">", "&gt;")
        content = content.replace("\n", "<br>")
        super().__init__("p", content, attributes)


class Link(BaseHTMLElement):
    def __init__(
        self,
        content: Union[str, BaseHTMLElement] = "",
        url: str = "",
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        attributes = dict(attributes or {})
        attributes["href"] = url
        super().__init__("a", content, attributes)


class Divider(BaseHTMLElement):
    def __init__(
        self,
        size: int = 1,
        color: str = "#e2e2e2",
        direction: Literal["horizontal", "vertical"] = "horizontal",
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        if direction == "horizontal":
            attributes = {**(attributes or {}), "color": color, "size": str(size)}
            super().__init__("hr", "", attributes)
        else:
            style = f"border-left: {size}px solid {color};"
            attributes = _update_style(attributes, style)
            super().__init__("div", "", attributes)


class Section(BaseHTMLElement):
    def __init__(
        self,
        *content: Union[str, BaseHTMLElement],
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        content_ = "".join(str(item) for item in content)
        super().__init__("section", content_, attributes)


class Table(BaseHTMLElement):
    @staticmethod
    def _build_content(
        rows: Sequence[Sequence[Union[str, BaseHTMLElement]]],
        header: Optional[Sequence[Union[str, BaseHTMLElement]]] = None,
        caption: Optional[Union[str, BaseHTMLElement]] = None,
    ) -> str:
        content = ""
        if caption:
            content += f"<caption>{caption}</caption>"

        if header:
            content += "<thead>"
            content += "<tr>"
            for cell in header:
                content += f"<th>{cell}</th>"
            content += "</tr>"
            content += "</thead>"

        content += "<tbody>"
        for row in rows:
            content += "<tr>"
            for cell in row:
                content += f"<td>{cell}</td>"
            content += "</tr>"
        content += "</tbody>"
        return content

    def __init__(
        self,
        rows: Sequence[Sequence[Union[str, BaseHTMLElement]]],
        header: Optional[Sequence[Union[str, BaseHTMLElement]]] = None,
        caption: Optional[Union[str, BaseHTMLElement]] = None,
        attributes: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(
            "table",
            self._build_content(rows, header, caption),
            attributes,
        )


class Image(BaseHTMLElement):
    def __init__(
        self,
        src: str,
        alt: str,
        attributes: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(
            "img",
            "",
            {
                **(attributes or {}),
                "src": src,
                "alt": alt,
            },
        )


class Plot(Image):
    @staticmethod
    def _build_base64(fig: "plt.Figure") -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        base64_string = base64.b64encode(buf.read()).decode("utf-8")
        return base64_string

    def __init__(
        self,
        fig: "plt.Figure",
        alt: str,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        if plt is None:
            raise ImportError("matplotlib is required for Plot elements")

        src = f"data:image/png;base64,{self._build_base64(fig)}"
        super().__init__(src, alt, attributes)


class Code(BaseHTMLElement):
    def __init__(
        self,
        content: str,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__("code", content, attributes)


class CodeBlock(BaseHTMLElement):
    def __init__(
        self,
        code: str,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(
            "pre",
            Code(code),
            attributes,
        )


_DEFAULT_META = [
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
]
_DEFAULT_STYLE = dedent(
    """
    body {
      width: fit-content;
      max-width: 1000px;
      padding: 0 10px;
      margin: 0 auto;
      font-family: 'Helvetica Neue', Arial, sans-serif;'
      font-size: 16px;
      line-height: 1.6;
      color: #333;
      background-color: #f8f9fa;
    }

    h1, h2, h3, h4, h5, h6 {
      margin: 5px 0;
    }

    pre {
      background-color: #f4f4f4;
      padding: 10px;
      border-radius: 10px;
    }

    table {
      width: 100%;
      height: fit-content;
      border-collapse: collapse;
      text-align: left;
      color: #333;
    }

    th {
      background-color: #f2f2f2;
      color: #333;
      padding: 7px 12px;
    }

    td {
      padding: 7px 12px;
    }

    tr:nth-child(even) {
      background-color: #f7f7f7;
    }

    tr:hover {
      background-color: #f1f1f1;
    }
    """
).strip()


def build_html(
    *content: Union[str, BaseHTMLElement],
    title: str = "HTML Builder",
    style: str = _DEFAULT_STYLE,
    meta: Union[str, Sequence[str]] = _DEFAULT_META,
    favicon: Optional[str] = None,
    lang: str = "en",
) -> str:
    output = "<!DOCTYPE html>"
    output += f"<html lang='{lang}'>"

    output += "<head>"
    output += f"<title>{title}</title>"
    output += f"<style>{style}</style>"
    if favicon:
        output += f"<link rel='icon' href='{favicon}' />"
    if meta:
        if isinstance(meta, str):
            meta = [meta]
        for tag in meta:
            output += tag
    output += "</head>"

    output += "<body>"
    content_ = "".join(str(c) for c in content)
    output += content_
    output += "</body>"

    output += "</html>"
    return output
