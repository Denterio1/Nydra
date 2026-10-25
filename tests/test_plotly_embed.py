def test_plotly_is_inlined():
    from src.vision_nlp import image_report as r
    tag = r._plotly_script_tag()
    assert tag.startswith("<script>")
    assert len(tag) > 1_000_000
    assert "Plotly" in tag