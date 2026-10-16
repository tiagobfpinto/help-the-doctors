param([switch]$Offline)

$taskRoot = $PSScriptRoot
$taskCompiler = Join-Path $taskRoot '.tools\tectonic\tectonic.exe'
if (-not (Test-Path -LiteralPath $taskCompiler)) {
    $taskInstalled = Get-Command tectonic -ErrorAction SilentlyContinue
    if (-not $taskInstalled) {
        throw 'Tectonic is missing. Run install_latex.ps1 first.'
    }
    $taskCompiler = $taskInstalled.Source
}
$taskOutput = Join-Path $taskRoot 'output\pdf'
$taskCache = Join-Path $taskRoot '.tools\tectonic\cache'
New-Item -ItemType Directory -Path $taskOutput, $taskCache -Force | Out-Null
$taskPreviousCache = $env:TECTONIC_CACHE_DIR
$taskPreviousFontconfig = $env:FONTCONFIG_FILE
$taskPreviousLocation = Get-Location
try {
    $env:TECTONIC_CACHE_DIR = $taskCache
    # Tectonic's portable Windows binary does not ship a default fontconfig
    # file. Keep that configuration and its cache inside this installation.
    $taskFontCache = Join-Path $taskRoot '.tools\tectonic\fontconfig-cache'
    New-Item -ItemType Directory -Path $taskFontCache -Force | Out-Null
    $taskFontConfig = Join-Path $taskRoot '.tools\tectonic\fonts.conf'
    $taskFontDirectory = [System.Security.SecurityElement]::Escape((Join-Path $env:WINDIR 'Fonts').Replace('\', '/'))
    $taskFontCacheXml = [System.Security.SecurityElement]::Escape($taskFontCache.Replace('\', '/'))
    $taskFontXml = "<?xml version=`"1.0`"?><fontconfig><dir>$taskFontDirectory</dir><cachedir>$taskFontCacheXml</cachedir></fontconfig>"
    [System.IO.File]::WriteAllText($taskFontConfig, $taskFontXml, (New-Object System.Text.UTF8Encoding($false)))
    $env:FONTCONFIG_FILE = $taskFontConfig
    Set-Location -LiteralPath $taskRoot
    $taskArguments = @('--keep-logs', '--keep-intermediates', '--outdir', $taskOutput, '--untrusted')
    if ($Offline) { $taskArguments += '--only-cached' }
    $taskArguments += (Join-Path $taskRoot 'paper.tex')
    & $taskCompiler @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Tectonic failed with exit code $LASTEXITCODE." }
    $taskLog = Get-Content -LiteralPath (Join-Path $taskOutput 'paper.log') -Raw
    $taskTexSource = Get-Content -LiteralPath (Join-Path $taskRoot 'paper.tex') -Raw
    if ($taskTexSource -match '\\bibliography\{') {
        $taskBibLog = Get-Content -LiteralPath (Join-Path $taskOutput 'paper.blg') -Raw
        if ($taskBibLog -match "I couldn't open|I found no|There (was|were) [0-9]+ error") {
            throw 'BibTeX failed; the PDF must contain all references before it can be packaged.'
        }
    }
    if ($taskLog -match 'There were undefined (citations|references)|Citation .+ undefined|No file paper\.bbl') {
        throw 'The PDF has unresolved citations or a missing bibliography.'
    }
    $taskPageMatch = [regex]::Match($taskLog, 'PAPER BODY PAGES=(\d+)')
    if (-not $taskPageMatch.Success) { throw 'The body-page count is missing from the compilation log.' }
    $taskBodyPages = [int]$taskPageMatch.Groups[1].Value
    $taskTotalMatch = [regex]::Match($taskLog, 'PAPER TOTAL PAGES=(\d+)')
    if (-not $taskTotalMatch.Success) { throw 'The total PDF page count is missing from the compilation log.' }
    $taskTotalPages = [int]$taskTotalMatch.Groups[1].Value
    if ($taskTotalPages -gt 3) { throw "The entire PDF has $taskTotalPages pages including references; the limit is three." }
    $taskSourceHashes = [ordered]@{}
    foreach ($taskSource in @('paper.tex', 'paper.bib', 'paper_assets/references.tex', 'paper_assets/results.tex', 'paper_assets/per_class.tex', 'paper_assets/confusion_matrix.png', 'selected_model.json')) {
        $taskSourceHashes[$taskSource] = (Get-FileHash -LiteralPath (Join-Path $taskRoot $taskSource) -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    [pscustomobject]@{
        compiler = (& $taskCompiler --version | Out-String).Trim()
        body_pages = $taskBodyPages
        total_pages = $taskTotalPages
        bibliography_excluded = $false
        page_limit_scope = 'entire PDF including bibliography'
        pdf = 'output/pdf/paper.pdf'
        paper_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskRoot 'paper.tex') -Algorithm SHA256).Hash.ToLowerInvariant()
        pdf_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskOutput 'paper.pdf') -Algorithm SHA256).Hash.ToLowerInvariant()
        source_sha256 = $taskSourceHashes
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'compilation.json') -Encoding UTF8
    Write-Output "Verified entire PDF: $taskTotalPages pages including bibliography. PDF: $taskOutput\paper.pdf"
} finally {
    Set-Location -LiteralPath $taskPreviousLocation.Path
    $env:TECTONIC_CACHE_DIR = $taskPreviousCache
    $env:FONTCONFIG_FILE = $taskPreviousFontconfig
}
