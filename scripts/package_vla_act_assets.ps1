param(
    [string]$SourceRoot,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $SourceRoot) {
    $SourceRoot = Join-Path $repoRoot "docs/VLA/ACT_for_microscopy"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "dist/vla_act_assets"
}

if (-not (Test-Path $SourceRoot)) {
    throw "Source directory does not exist: $SourceRoot"
}

$assetSpecs = @(
    @{ RelativePath = "README.md"; AssetName = "ACT_for_microscopy_root.zip"; Type = "file" },
    @{ RelativePath = "2D_move_none"; AssetName = "ACT_for_microscopy_2D_move_none.zip"; Type = "dir" },
    @{ RelativePath = "2D_set_brightness_none"; AssetName = "ACT_for_microscopy_2D_set_brightness_none.zip"; Type = "dir" },
    @{ RelativePath = "2D_set_z_funa"; AssetName = "ACT_for_microscopy_2D_set_z_funa.zip"; Type = "dir" },
    @{ RelativePath = "2D_set_z_none"; AssetName = "ACT_for_microscopy_2D_set_z_none.zip"; Type = "dir" },
    @{ RelativePath = "Cell_move_none"; AssetName = "ACT_for_microscopy_Cell_move_none.zip"; Type = "dir" },
    @{ RelativePath = "Cell_set_brightness_none"; AssetName = "ACT_for_microscopy_Cell_set_brightness_none.zip"; Type = "dir" },
    @{ RelativePath = "Cell_set_z_funa"; AssetName = "ACT_for_microscopy_Cell_set_z_funa.zip"; Type = "dir" },
    @{ RelativePath = "Cell_set_z_none"; AssetName = "ACT_for_microscopy_Cell_set_z_none.zip"; Type = "dir" },
    @{ RelativePath = "Push_to target"; AssetName = "ACT_for_microscopy_Push_to_target.zip"; Type = "dir" },
    @{ RelativePath = "Slice_move_none"; AssetName = "ACT_for_microscopy_Slice_move_none.zip"; Type = "dir" },
    @{ RelativePath = "Slice_set_brightness_none"; AssetName = "ACT_for_microscopy_Slice_set_brightness_none.zip"; Type = "dir" },
    @{ RelativePath = "Slice_set_z_fbna"; AssetName = "ACT_for_microscopy_Slice_set_z_fbna.zip"; Type = "dir" },
    @{ RelativePath = "Slice_set_z_none"; AssetName = "ACT_for_microscopy_Slice_set_z_none.zip"; Type = "dir" }
)

$stagingRoot = Join-Path $OutputDir "_staging"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($spec in $assetSpecs) {
    $sourcePath = Join-Path $SourceRoot $spec.RelativePath
    if (-not (Test-Path $sourcePath)) {
        throw "Missing source path: $sourcePath"
    }

    if (Test-Path $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }

    $stageContainer = Join-Path $stagingRoot "ACT_for_microscopy"
    New-Item -ItemType Directory -Force -Path $stageContainer | Out-Null

    if ($spec.Type -eq "dir") {
        Copy-Item -LiteralPath $sourcePath -Destination $stageContainer -Recurse -Force
    }
    else {
        Copy-Item -LiteralPath $sourcePath -Destination $stageContainer -Force
    }

    $zipPath = Join-Path $OutputDir $spec.AssetName
    if (Test-Path $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    Compress-Archive -LiteralPath (Join-Path $stagingRoot "ACT_for_microscopy") -DestinationPath $zipPath -CompressionLevel Optimal
}

if (Test-Path $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}

Write-Host "Created VLA ACT asset packages in $OutputDir"
Get-ChildItem $OutputDir -Filter *.zip | Select-Object Name, Length
