"""Regression tests for DOCXReader.

Bug this guards against: _extract_core_props used `cp.description`, which
python-docx does not have, so every .docx failed with AttributeError.
"""
import pytest
import pathlib as Path
docx = pytest.importorskip("docx")

from src.vision_nlp.document_reader import DOCXReader


@pytest.fixture
def sample_docx(tmp_path):
    doc = docx.Document()
    doc.core_properties.author = "Test Author"
    doc.core_properties.title = "Test Title"
    doc.core_properties.comments = "A short description"
    doc.add_heading("Quarterly Report", level=1)
    doc.add_paragraph("Revenue grew steadily across all regions this quarter.")
    doc.add_paragraph("Costs stayed flat while headcount increased slightly.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "region"
    table.cell(0, 1).text = "revenue"
    table.cell(1, 0).text = "north"
    table.cell(1, 1).text = "100"
    path = tmp_path / "sample.docx"
    doc.save(str(path))
    return str(path)


@pytest.fixture
def empty_docx(tmp_path: Path):
    path = tmp_path / "empty.docx"
    d = docx.Document()
    d.core_properties.comments = ""
    d.save(str(path))
    return str(path)


def test_read_does_not_raise_on_real_docx(sample_docx):
    # The original bug: AttributeError inside _extract_core_props.
    data = DOCXReader().read(sample_docx)
    assert data.fmt == "docx"
    assert data.file_path == sample_docx


def test_read_extracts_text(sample_docx):
    data = DOCXReader().read(sample_docx)
    assert "Revenue grew steadily" in data.full_text
    assert "Quarterly Report" in data.full_text
    assert data.word_count > 0


def test_read_extracts_table(sample_docx):
    data = DOCXReader().read(sample_docx)
    assert len(data.tables) >= 1


def test_core_props_has_description_from_comments(sample_docx):
    reader = DOCXReader()
    props = reader._extract_core_props(docx.Document(sample_docx))
    assert props["description"] == "A short description"
    assert props["author"] == "Test Author"
    assert props["title"] == "Test Title"


def test_core_props_description_defaults_to_empty_string(empty_docx):
    props = DOCXReader()._extract_core_props(docx.Document(empty_docx))
    assert props["description"] == ""


def test_read_empty_docx_does_not_raise(empty_docx):
    data = DOCXReader().read(empty_docx)
    assert data.fmt == "docx"
    assert data.word_count == 0