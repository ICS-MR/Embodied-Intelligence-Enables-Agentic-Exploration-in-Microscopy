param(
    [string]$Repo = $(if ($env:EIMS_DETECTOR_WEIGHTS_REPO) { $env:EIMS_DETECTOR_WEIGHTS_REPO } else { "ICS-MR/Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy" }),
    [string]$ReleaseTag = $(if ($env:EIMS_DETECTOR_WEIGHTS_RELEASE_TAG) { $env:EIMS_DETECTOR_WEIGHTS_RELEASE_TAG } else { "detector-weights" }),
    [string]$AssetBaseUrl = $env:EIMS_DETECTOR_WEIGHTS_BASE_URL,
    [string]$TargetRoot,
    [string]$AssetDir,
    [ValidateSet("2Dcell", "organoid", "mitosis")]
    [string[]]$Models = @("2Dcell", "organoid", "mitosis"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $TargetRoot) {
    $TargetRoot = Join-Path $repoRoot "weights"
}

$downloadDir = if ($AssetDir) { $AssetDir } else { Join-Path $repoRoot ".runtime/downloads/detector-weights" }

$modelAssets = @{
    "2Dcell" = @{
        AssetName = "2Dcell.pth"
        RelativePath = "2Dcell.pth"
    }
    "organoid" = @{
        AssetName = "organoid.pth"
        RelativePath = "organoid.pth"
    }
    "mitosis" = @{
        AssetName = "mitosis_best.pth"
        RelativePath = "mitosis_best.pth"
    }
}

function Get-AssetUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AssetName
    )

    if ($AssetBaseUrl) {
        return ($AssetBaseUrl.TrimEnd("/") + "/" + $AssetName)
    }

    return "https://github.com/$Repo/releases/download/$ReleaseTag/$AssetName"
}

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

foreach ($modelName in $Models) {
    if (-not $modelAssets.ContainsKey($modelName)) {
        throw "Unsupported detector weight group: $modelName"
    }

    $asset = $modelAssets[$modelName]
    $assetName = $asset.AssetName
    $targetPath = Join-Path $TargetRoot $asset.RelativePath
    $localDownloadPath = Join-Path $downloadDir $assetName

    if ((Test-Path $targetPath) -and (-not $Force)) {
        Write-Host "Using existing detector weight: $targetPath"
        continue
    }

    $assetUrl = Get-AssetUrl -AssetName $assetName
    Write-Host "Downloading $assetName from release '$ReleaseTag' ..."
    Invoke-WebRequest -Uri $assetUrl -OutFile $localDownloadPath

    Copy-Item -LiteralPath $localDownloadPath -Destination $targetPath -Force
    Write-Host "Installed detector weight to $targetPath"
}
