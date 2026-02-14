$initVersion = (Select-String -Path 'empulse/__init__.py' -Pattern "__version__ = '([^']+)'").Matches.Groups[1].Value
$citationVersion = (Select-String -Path 'CITATION.cff' -Pattern '^version: (.+)').Matches.Groups[1].Value.Trim()
$changelogContent = Get-Content 'CHANGELOG.rst' -Raw

if ($initVersion -ne $citationVersion) {
    Write-Host "Version mismatch: __init__.py ($initVersion) != CITATION.cff ($citationVersion)" -ForegroundColor Red
    exit 1
}

if (-not ($changelogContent -match [regex]::Escape($initVersion))) {
    Write-Host "Version $initVersion not found in CHANGELOG.rst" -ForegroundColor Red
    exit 1
}

Write-Host "Version $initVersion is consistent across all files" -ForegroundColor Green