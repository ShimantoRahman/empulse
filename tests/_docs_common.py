import re
import textwrap

# Matches the start of a Python code block at any indentation, so blocks nested inside a directive
# (a ``tab-item``, for instance, or a method body) are found too.
CODE_BLOCK_START_RE = re.compile(r'^(?P<indent>[ \t]*)\.\. code-block:: python\s*$')

# Directive options such as ``:caption:`` or ``:linenos:`` sit between the directive and its body.
DIRECTIVE_OPTION_RE = re.compile(r'^[ \t]*:[\w-]+:.*$')


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks, including any nested inside another directive.

    A block's body is every following line indented further than the directive itself, so the end
    of a block is found by indentation rather than by looking for the next line in column zero. The
    naive version of this stopped at the first unindented line, which silently swallowed the prose
    after a nested block and then failed to compile it.

    Shared between ``tests/test_user_guide.py`` (run against ``.rst`` file contents) and
    ``tests/test_docstring.py`` (run against docstrings, which use the same directive syntax).
    """
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        match = CODE_BLOCK_START_RE.match(lines[index])
        if match is None:
            index += 1
            continue

        indent = len(match.group('indent').expandtabs(8))
        index += 1

        # Skip the directive's own options and the blank line separating them from the body.
        while index < len(lines) and (DIRECTIVE_OPTION_RE.match(lines[index]) or not lines[index].strip()):
            index += 1

        body = []
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                body.append('')  # a blank line does not end the block
                index += 1
                continue
            if len(line.expandtabs(8)) - len(line.expandtabs(8).lstrip()) <= indent:
                break
            body.append(line)
            index += 1

        if body:
            blocks.append(textwrap.dedent('\n'.join(body).rstrip()))
    return blocks
