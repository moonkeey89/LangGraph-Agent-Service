from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".pdf"})


class KnowledgeDocumentError(ValueError):
    """A safe, user-facing failure while loading a local source document."""


class Utf8TextDocumentLoader(BaseLoader):
    def __init__(self, path: Path):
        self.path = path

    def lazy_load(self) -> Iterator[Document]:
        try:
            content = self.path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise KnowledgeDocumentError(
                f"文本文件不是有效的UTF-8编码：{self.path.name}"
            ) from error
        except OSError as error:
            raise KnowledgeDocumentError(
                f"无法读取文本文件：{self.path.name}"
            ) from error
        if not content.strip():
            raise KnowledgeDocumentError(f"文档内容为空：{self.path.name}")
        yield Document(
            page_content=content,
            metadata={"source": self.path.name},
        )


class LocalPdfDocumentLoader(BaseLoader):
    def __init__(self, path: Path):
        self.path = path

    def lazy_load(self) -> Iterator[Document]:
        try:
            reader = PdfReader(str(self.path))
            if reader.is_encrypted:
                raise KnowledgeDocumentError(
                    f"暂不支持加密PDF：{self.path.name}"
                )
            extracted = False
            for page_number, page in enumerate(reader.pages, start=1):
                content = page.extract_text() or ""
                if not content.strip():
                    continue
                extracted = True
                yield Document(
                    page_content=content,
                    metadata={
                        "source": self.path.name,
                        "page": page_number,
                    },
                )
            if not extracted:
                raise KnowledgeDocumentError(
                    f"PDF没有可提取文本，可能是扫描件：{self.path.name}"
                )
        except KnowledgeDocumentError:
            raise
        except Exception as error:
            raise KnowledgeDocumentError(
                f"PDF损坏或无法解析：{self.path.name}"
            ) from error


def loader_for(path: Path) -> BaseLoader:
    if not path.exists() or not path.is_file():
        raise KnowledgeDocumentError(f"文档不存在或不是文件：{path}")
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md", ".markdown"}:
        return Utf8TextDocumentLoader(path)
    if suffix == ".pdf":
        return LocalPdfDocumentLoader(path)
    raise KnowledgeDocumentError(
        f"不支持的文档格式：{path.suffix or '<无扩展名>'}；"
        "仅支持TXT、Markdown和PDF"
    )


def discover_documents(paths: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for input_path in paths:
        path = input_path.expanduser().resolve()
        if path.is_dir():
            discovered.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.casefold() in SUPPORTED_SUFFIXES
            )
        elif path.is_file():
            discovered.add(path)
        else:
            raise KnowledgeDocumentError(f"路径不存在：{input_path}")
    if not discovered:
        raise KnowledgeDocumentError("没有发现可入库的TXT、Markdown或PDF文档")
    return sorted(discovered)
