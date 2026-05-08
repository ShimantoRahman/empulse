import re
import subprocess
import sys


def main():
    """Find tox environments and run them with tox-uv."""
    with open('tox.ini', encoding='utf-8') as f:
        content = f.read()
    envs = re.findall(r'\[testenv:(sklearn\d+-tests)\]', content)

    result = subprocess.run(['uvx', '--with', 'tox-uv', 'tox', '-e', ','.join(envs)], check=False)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
