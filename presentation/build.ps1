[CmdletBinding()]
param(
    [ValidateSet('presentation_labs_2_3', 'presentation_labs_4_5')]
    [string[]]$Deck = @('presentation_labs_2_3', 'presentation_labs_4_5')
)

$ErrorActionPreference = 'Stop'
$compiler = (Get-Command pdflatex -ErrorAction Stop).Source
$buildRoot = Join-Path $PSScriptRoot '.build'
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
Push-Location $PSScriptRoot
try {
    foreach ($name in $Deck) {
        $outDir = Join-Path $buildRoot $name
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        for ($pass = 1; $pass -le 2; $pass++) {
            & $compiler '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' "-output-directory=$outDir" "$name.tex"
            if ($LASTEXITCODE -ne 0) {
                throw "LaTeX failed: $name, pass $pass. See $outDir\$name.log"
            }
        }
        Copy-Item -LiteralPath (Join-Path $outDir "$name.pdf") -Destination (Join-Path $PSScriptRoot "$name.pdf") -Force
        Write-Host "Built $name.pdf (two passes)"
    }
}
finally {
    Pop-Location
}
