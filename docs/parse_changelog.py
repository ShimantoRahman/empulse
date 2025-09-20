import re
import sys
import argparse

DOC_BASE = "https://empulse.readthedocs.io/en/stable/reference/"

def extract_changelog_section(changelog, version):
    pattern = (
        rf"(?ms)^`{re.escape(version)}`_.*?^=+\r?\n(.*?)(?=^`[0-9]+\.[0-9]+\.[0-9]+[a-z]*[0-9]*`_|\Z)"
    )
    match = re.search(pattern, changelog)
    return match.group(1).strip() if match else None

def replace_custom_tags(text):
    replacements = {
        r"\|MajorFeature\|": "![Major Feature](https://img.shields.io/badge/-Major%20Feature-forestgreen)",
        r"\|Feature\|": "![Feature](https://img.shields.io/badge/-Feature-forestgreen)",
        r"\|Enhancement\|": "![Enhancement](https://img.shields.io/badge/-Enhancement-deepskyblue)",
        r"\|Efficiency\|": "![Efficiency](https://img.shields.io/badge/-Efficiency-deepskyblue)",
        r"\|Fix\|": "![Fix](https://img.shields.io/badge/-Fix-red)",
        r"\|API\|": "![API Change](https://img.shields.io/badge/-API%20Change-yellow)",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def rst_role_to_md_link(match):
    role = match.group(1)
    obj = match.group(2).lstrip("~")
    parts = obj.split(".")
    if role == "meth":
        html_file = ".".join(parts[:-1]) + ".html"
        anchor = obj
    elif role == "mod":
        html_file = parts[-1] + ".html"
        anchor = parts[-1]
    else:
        html_file = obj + ".html"
        anchor = obj
    link_text = parts[-1]
    if role == "mod":
        url = f"{DOC_BASE}{html_file}"
    elif role == "ref":
        return "[User Guide](https://empulse.readthedocs.io/en/stable/guide.html)"
    else:
        url = f"{DOC_BASE}generated/{html_file}#{anchor}"
    return f"[`{link_text}`]({url})"

def convert_rst_roles(text):
    # Handles :class:`~empulse.metrics.Metric`, :meth:, :func:, :attr:, etc.
    pattern = r":(class|meth|func|attr|mod|ref):`([^`]+)`"
    return re.sub(pattern, rst_role_to_md_link, text)

def main():
    parser = argparse.ArgumentParser(description="Extract and convert CHANGELOG.rst section to Markdown.")
    parser.add_argument("version", help="Version to extract (e.g. 0.9.0)")
    parser.add_argument("-i", "--input", default="CHANGELOG.rst", help="Input changelog file (default: CHANGELOG.rst)")
    parser.add_argument("-o", "--output", default="section.md", help="Output markdown file (default: section.md)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        changelog = f.read()

    section = extract_changelog_section(changelog, args.version)
    if not section:
        print(f"Version {args.version} not found in {args.input}")
        sys.exit(1)

    markdown = replace_custom_tags(section)
    markdown = convert_rst_roles(markdown)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Section for version {args.version} written to {args.output}")

if __name__ == "__main__":
    main()