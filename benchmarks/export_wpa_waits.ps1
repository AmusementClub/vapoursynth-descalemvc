[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Etl,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [string]$Prefix = "waits_",
    [ValidateSet("Detailed", "Grouped", "Reasons", "ReadyStack", "NewStack")]
    [string]$Mode = "Detailed",
    [ValidateRange(1, 32)]
    [int]$ExpansionDepth = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Etl = [IO.Path]::GetFullPath($Etl)
$Output = [IO.Path]::GetFullPath($Output)
$Toolkit = Join-Path ([Environment]::GetFolderPath("ProgramFilesX86")) `
    "Windows Kits\10\Windows Performance Toolkit"
$Exporter = Join-Path $Toolkit "wpaexporter.exe"
$SourceProfile = Join-Path $Toolkit "Catalog\AppLaunch.wpaProfile"
foreach ($path in @($Etl, $Exporter, $SourceProfile)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "required file does not exist: $path"
    }
}
New-Item -ItemType Directory -Path $Output -Force | Out-Null

$document = [Xml.XmlDocument]::new()
$document.PreserveWhitespace = $true
$document.Load($SourceProfile)
$namespace = [Xml.XmlNamespaceManager]::new($document.NameTable)
$namespace.AddNamespace("w", "http://tempuri.org/SerializableElement.xsd")
$preciseGuid = "c58f5fea-0319-4046-932d-e695ebe20b47"
$graph = $document.SelectSingleNode(
    "//w:Views/w:View/w:Graphs/w:Graph[@Guid='$preciseGuid']", $namespace)
$threadDelays = $document.SelectSingleNode(
    "//w:ModifiedGraphs/w:GraphSchema[@Guid='$preciseGuid']" +
    "/w:ModifiedPresets/w:Preset[@Name='Thread Delays']", $namespace)
if (-not $graph -or -not $threadDelays) {
    throw "the installed AppLaunch profile has no CPU Precise Thread Delays preset"
}

$graphs = $graph.ParentNode
foreach ($candidate in @($graphs.ChildNodes)) {
    if ($candidate -ne $graph) {
        [void]$graphs.RemoveChild($candidate)
    }
}
foreach ($preset in @($graph.SelectNodes("w:Preset", $namespace))) {
    [void]$graph.RemoveChild($preset)
}
$expandedPreset = $document.ImportNode($threadDelays, $true)
$expandedPreset.SetAttribute(
    "InitialFilterQuery", '[New Process]:~="VSPipe.exe"')
$expandedPreset.SetAttribute("InitialFilterShouldKeep", "true")
if ($Mode -ne "Detailed") {
    $sortColumns = switch ($Mode) {
        "Grouped" {
            @(
                "New Process", "New Thread Id", "New Prev State",
                "New Prev Wait Reason", "New Prev Wait Mode",
                "New Thread Stack", "Readying Process", "Readying Thread Id",
                "Ready Thread Stack"
            )
        }
        "Reasons" {
            @(
                "New Process", "New Thread Id", "New Prev State",
                "New Prev Wait Reason", "New Prev Wait Mode",
                "Readying Process"
            )
        }
        "ReadyStack" {
            @(
                "New Process", "Readying Process", "Readying Thread Id",
                "Ready Thread Stack", "New Prev Wait Reason"
            )
        }
        "NewStack" {
            @(
                "New Process", "New Thread Id", "New Thread Stack",
                "New Prev Wait Reason"
            )
        }
    }
    $columns = @($expandedPreset.SelectNodes("w:Columns/w:Column", $namespace))
    foreach ($column in $columns) {
        $column.RemoveAttribute("SortPriority")
        $column.RemoveAttribute("SortOrder")
    }
    for ($index = 0; $index -lt $sortColumns.Count; ++$index) {
        $name = $sortColumns[$index]
        $column = $columns | Where-Object { $_.GetAttribute("Name") -eq $name } |
            Select-Object -First 1
        if (-not $column) { throw "Thread Delays preset has no '$name' column" }
        $column.SetAttribute("SortOrder", "Ascending")
        if ($index -gt 0) {
            $column.SetAttribute("SortPriority", "$index")
        }
        $column.SetAttribute("IsVisible", "true")
    }
    $expandedPreset.SetAttribute("KeyColumnCount", "$($sortColumns.Count)")
    $ExpansionDepth = $sortColumns.Count
}
$expandedPreset.SetAttribute(
    "InitialExpansionQuery", "[Series Depth]:<=$ExpansionDepth")
[void]$graph.AppendChild($expandedPreset)

$fileReferences = $document.SelectSingleNode(
    "//w:Sessions/w:Session/w:FileReferences", $namespace)
if ($fileReferences) {
    [void]$fileReferences.ParentNode.RemoveChild($fileReferences)
}
$modifiedGraphs = $document.SelectSingleNode("//w:ModifiedGraphs", $namespace)
if ($modifiedGraphs) {
    [void]$modifiedGraphs.ParentNode.RemoveChild($modifiedGraphs)
}

$profile = Join-Path $Output "cpu-precise-thread-delays.wpaProfile"
$document.Save($profile)
$command = @(
    "-i", $Etl,
    "-profile", $profile,
    "-outputfolder", $Output,
    "-prefix", $Prefix,
    "-outputformat", "CSV"
)
$log = Join-Path $Output "$($Prefix.TrimEnd('_'))-wpaexporter.log"
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $lines = & $Exporter @command 2>&1
    $exitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorAction
}
$lines | Set-Content -LiteralPath $log -Encoding UTF8
if ($exitCode -ne 0) {
    throw "wpaexporter failed with exit code $exitCode; see $log"
}

$csv = Get-ChildItem -LiteralPath $Output -Filter "$Prefix*.csv" |
    Where-Object { $_.Name -match 'CPU_Usage_.*Precise.*Thread_Delays' } |
    Select-Object -First 1
if (-not $csv -or $csv.Length -eq 0) {
    throw "wpaexporter did not create a nonempty CPU Precise Thread Delays CSV"
}
$header = Get-Content -LiteralPath $csv.FullName -TotalCount 1
if ($header -notmatch '^New Process,New Thread Id,') {
    throw "unexpected Thread Delays CSV header: $header"
}

[pscustomobject]@{
    etl = $Etl
    etl_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Etl).Hash
    profile = $profile
    profile_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $profile).Hash
    csv = $csv.FullName
    csv_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $csv.FullName).Hash
    csv_bytes = $csv.Length
    exporter = $Exporter
    exporter_version = (Get-Item -LiteralPath $Exporter).VersionInfo.FileVersion
    mode = $Mode
    expansion_depth = $ExpansionDepth
} | ConvertTo-Json | Set-Content -LiteralPath `
    (Join-Path $Output "$($Prefix.TrimEnd('_'))-manifest.json") -Encoding UTF8

Write-Output $csv.FullName
