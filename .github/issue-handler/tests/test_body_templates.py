"""Unit tests for body_templates: parse_sections and update_section."""
from issue_handler.utils.body_templates import parse_sections, update_section


SAMPLE_BODY = """\
<!-- agent:status:DISCOVERED -->

## Summary
This is a summary.

## Failed Tests
test_foo
test_bar

## Root Cause Analysis

## Action Items
- [ ] Issue formatted
"""


class TestParseSections:
    def test_basic_sections(self):
        sections = parse_sections(SAMPLE_BODY)
        assert "Summary" in sections
        assert sections["Summary"] == "This is a summary."

    def test_multiline_section(self):
        sections = parse_sections(SAMPLE_BODY)
        assert "test_foo" in sections["Failed Tests"]
        assert "test_bar" in sections["Failed Tests"]

    def test_empty_section(self):
        sections = parse_sections(SAMPLE_BODY)
        assert sections.get("Root Cause Analysis", "") == ""

    def test_h3_headings(self):
        body = "### Sub Section\ncontent here\n\n### Another\nmore"
        sections = parse_sections(body)
        assert sections["Sub Section"] == "content here"
        assert sections["Another"] == "more"

    def test_no_sections(self):
        sections = parse_sections("Just plain text\nno headings")
        assert sections == {}


class TestUpdateSection:
    def test_replace_existing(self):
        result = update_section(SAMPLE_BODY, "Root Cause Analysis", "Memory leak in foo()")
        assert "Memory leak in foo()" in result
        # Heading preserved
        assert "## Root Cause Analysis" in result

    def test_preserves_other_sections(self):
        result = update_section(SAMPLE_BODY, "Root Cause Analysis", "Fixed")
        assert "This is a summary." in result
        assert "test_foo" in result

    def test_creates_missing_section_before_action_items(self):
        result = update_section(SAMPLE_BODY, "Proposed Fix Strategy", "Patch line 42")
        assert "## Proposed Fix Strategy" in result
        assert "Patch line 42" in result
        # Should appear before Action Items
        fix_pos = result.index("Proposed Fix Strategy")
        action_pos = result.index("Action Items")
        assert fix_pos < action_pos

    def test_creates_section_at_end_if_no_action_items(self):
        body = "## Summary\nHello\n"
        result = update_section(body, "New Section", "content")
        assert "## New Section\ncontent" in result

    def test_h4_heading_match(self):
        """Regex matches #### headings too."""
        body = "#### Deep Section\nold content\n\n## Other\nstuff"
        result = update_section(body, "Deep Section", "new content")
        assert "new content" in result
        assert "old content" not in result
