$taskVersion = '0.17.0'
$taskExpectedHash = 'f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f'
$taskDirectory = Join-Path $PSScriptRoot '.tools\tectonic'
$taskFilename = "tectonic-$taskVersion-x86_64-pc-windows-msvc.zip"
$taskUrl = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40$taskVersion/$taskFilename"
New-Item -ItemType Directory -Path $taskDirectory -Force | Out-Null
$taskArchive = Join-Path $taskDirectory $taskFilename
if (-not (Test-Path -LiteralPath $taskArchive)) {
    Invoke-WebRequest -UseBasicParsing -Uri $taskUrl -OutFile $taskArchive -ErrorAction Stop
}
$taskActualHash = (Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskActualHash -ne $taskExpectedHash) { throw 'The compiler archive does not match the verified release checksum.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskZip = [System.IO.Compression.ZipFile]::OpenRead($taskArchive)
try {
    $taskBoundary = [System.IO.Path]::GetFullPath($taskDirectory) + [System.IO.Path]::DirectorySeparatorChar
    foreach ($taskEntry in $taskZip.Entries) {
        $taskResolved = [System.IO.Path]::GetFullPath((Join-Path $taskDirectory $taskEntry.FullName))
        if (-not $taskResolved.StartsWith($taskBoundary, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'The archive contains a path outside the installation folder.'
        }
    }
} finally { $taskZip.Dispose() }
Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskDirectory -Force -ErrorAction Stop
& (Join-Path $taskDirectory 'tectonic.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'The installed compiler could not run.' }
