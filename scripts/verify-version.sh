#!/bin/sh
init_version=$(sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" empulse/__init__.py)
citation_version=$(sed -n 's/^version: *//p' CITATION.cff | tr -d '[:space:]')

if [ "$init_version" != "$citation_version" ]; then
    printf '\033[31mVersion mismatch: __init__.py (%s) != CITATION.cff (%s)\033[0m\n' "$init_version" "$citation_version"
    exit 1
fi

if ! grep -qF "$init_version" CHANGELOG.rst; then
    printf '\033[31mVersion %s not found in CHANGELOG.rst\033[0m\n' "$init_version"
    exit 1
fi

printf '\033[32mVersion %s is consistent across all files\033[0m\n' "$init_version"
