[CmdletBinding()]
param(
    [string]$Output = "",
    [int]$Threads = 32,
    [int]$KernelFrames = 8192,
    [int]$SweepFrames = 500,
    [switch]$Elevated
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $Output) {
    $Output = Join-Path $RepoRoot `
        "benchmark-results\profile-current-4917f7b8-final"
}
$Output = [IO.Path]::GetFullPath($Output)
$CompletionRoot = Join-Path $Output "completion-profile"
$BinaryDir = Join-Path $Output "binary"
$RuntimeDir = Join-Path $Output "isolated-vs"
$SnapshotDir = Join-Path $Output "snapshot"
$Plugin = Join-Path $BinaryDir "dsmvc.dll"
$Pdb = Join-Path $BinaryDir "dsmvc.pdb"
$VSPipe = Join-Path $RuntimeDir "VSPipe.exe"
$KernelScript = Join-Path $SnapshotDir "vspipe_benchmark.vpy"
$SweepScript = Join-Path $SnapshotDir "vspipe_getnative.vpy"
$Image = "C:\Users\lsy39\Downloads\6.2-1.png"
$OldPlugin = `
    "D:\okegui\OKEGui\tools\vapoursynth\vapoursynth64\plugins\descale.dll"
$UProfCli = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"

$ExpectedPluginHash = `
    "893E41188AE707716FB856EB6663EF9E42C2722B82F157EA746F5F16E14C9001"
$ExpectedPdbHash = `
    "44538C0AE64F92E8CEA8C2162E03CC4ED85F8A8E494F7799956B9C45322DC2FA"
$ExpectedImageHash = `
    "61F9EE1AC858BBADD6A959BA35F5ECEB077B8452B91E97A5CE3D39EBC69E20C6"
$ExpectedOldPluginHash = `
    "B02E4A2FBAAF6BA3F7E3CF2AD8A08D8EEFAB9E5D634E1D829764671D49933000"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Escape-SingleQuoted([string]$Value) {
    return $Value.Replace("'", "''")
}

function Format-Command([string]$Executable, [string[]]$Arguments) {
    $quoted = foreach ($argument in $Arguments) {
        if ($argument -match '[\s"`$]') {
            "'$(Escape-SingleQuoted $argument)'"
        } else {
            $argument
        }
    }
    return "& '$(Escape-SingleQuoted $Executable)' $($quoted -join ' ')"
}

function Assert-FileHash(
    [string]$Path,
    [string]$Expected,
    [string]$Description
) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description does not exist: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    if ($actual -ne $Expected) {
        throw "$Description SHA-256 mismatch: expected $Expected, got $actual"
    }
}

Assert-FileHash $Plugin $ExpectedPluginHash "profiled plugin"
Assert-FileHash $Pdb $ExpectedPdbHash "profiled symbols"
Assert-FileHash $Image $ExpectedImageHash "benchmark input"
Assert-FileHash $OldPlugin $ExpectedOldPluginHash "baseline plugin"
foreach ($path in @($VSPipe, $KernelScript, $SweepScript, $UProfCli)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "required file does not exist: $path"
    }
}

New-Item -ItemType Directory -Path $CompletionRoot -Force | Out-Null
$MasterLog = Join-Path $CompletionRoot "completion-run.log"
$CommandLog = Join-Path $CompletionRoot "commands.ps1"
$ManifestPath = Join-Path $CompletionRoot "completion-manifest.json"

function Write-RunLog([string]$Message) {
    $line = "{0:o} {1}" -f [DateTimeOffset]::Now, $Message
    $line | Tee-Object -FilePath $MasterLog -Append | Write-Host
}

function Invoke-External(
    [string]$Executable,
    [string[]]$Arguments,
    [string]$LogPath,
    [switch]$AllowFailure
) {
    $command = Format-Command $Executable $Arguments
    Add-Content -LiteralPath $CommandLog -Value $command -Encoding UTF8
    Write-RunLog $command
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $lines = & $Executable @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    $lines | Set-Content -LiteralPath $LogPath -Encoding UTF8
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "$Executable failed with exit code $exitCode; see $LogPath"
    }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = $lines }
}

function New-VSPipeArguments(
    [string]$Case,
    [int]$Frames,
    [int]$Requests,
    [string]$Script
) {
    return @(
        "--arg", "implementation=new",
        "--arg", "plugin=$Plugin",
        "--arg", "old_plugin=$OldPlugin",
        "--arg", "image=$Image",
        "--arg", "case=$Case",
        "--arg", "frames=$Frames",
        "--arg", "threads=$Threads",
        "--requests", "$Requests",
        "--end", "$($Frames - 1)",
        "--filter-time", $Script, "."
    )
}

# Validate the frozen runtime before requesting elevation. This also makes a
# declined UAC prompt distinguishable from a bad plugin/runtime snapshot.
if (-not $Elevated) {
    $smokeArgs = New-VSPipeArguments "bicubic_b3" 2 1 $KernelScript
    Invoke-External $VSPipe $smokeArgs `
        (Join-Path $CompletionRoot "pre-elevation-smoke.log") | Out-Null

    $hostExe = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    $worker = @(
        "& '$(Escape-SingleQuoted $PSCommandPath)'",
        "-Elevated",
        "-Output '$(Escape-SingleQuoted $Output)'",
        "-Threads $Threads",
        "-KernelFrames $KernelFrames",
        "-SweepFrames $SweepFrames"
    ) -join " "
    $bootstrapLog = Join-Path $CompletionRoot "elevated-bootstrap.log"
    $invoke = @"
`$ErrorActionPreference = 'Stop'
try {
    $worker *>&1 | ForEach-Object {
        `$_ | Out-File -LiteralPath '$(Escape-SingleQuoted $bootstrapLog)' -Append -Encoding utf8
        Write-Host `$_
    }
    if (`$LASTEXITCODE -ne 0) {
        exit `$LASTEXITCODE
    }
    exit 0
} catch {
    (`$_ | Format-List * -Force | Out-String) | Out-File -LiteralPath '$(Escape-SingleQuoted $bootstrapLog)' -Append -Encoding utf8
    exit 1
}
"@
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($invoke))
    Write-Host "Requesting one elevated PowerShell worker for all IBS sessions."
    $process = Start-Process -FilePath $hostExe -Verb RunAs `
        -WindowStyle Hidden -ArgumentList `
        "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encoded" `
        -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "elevated IBS worker failed; see $bootstrapLog"
    }
    Write-Host "IBS completion worker finished successfully."
    exit 0
}

if (-not (Test-Administrator)) {
    throw "the IBS worker is not running as administrator"
}

$startedUtc = [DateTimeOffset]::UtcNow.ToString("o")
$sessions = [Collections.ArrayList]::new()
$manifest = [ordered]@{
    schema_version = 1
    purpose = "IBS and cache-tier completion pass"
    started_utc = $startedUtc
    completed_utc = $null
    elevated = $true
    git_commit = "4917f7b8306e89605d3986cf31583a869002a807"
    threads = $Threads
    kernel_frames = $KernelFrames
    sweep_frames = $SweepFrames
    cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores,
        NumberOfLogicalProcessors, MaxClockSpeed
    operating_system = Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption, Version, BuildNumber, TotalVisibleMemorySize
    plugin = $Plugin
    plugin_sha256 = $ExpectedPluginHash
    pdb = $Pdb
    pdb_sha256 = $ExpectedPdbHash
    old_plugin = $OldPlugin
    old_plugin_sha256 = $ExpectedOldPluginHash
    image = $Image
    image_sha256 = $ExpectedImageHash
    vspipe = $VSPipe
    vspipe_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $VSPipe).Hash
    kernel_script_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $KernelScript).Hash
    sweep_script_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $SweepScript).Hash
    uprof = $UProfCli
    ibs_scope = "system-wide"
    ibs_cpu_mask = $null
    ibs_call_graph = $false
    sessions = $sessions
}

function Save-Manifest {
    $manifest | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}

function Add-Session(
    [string]$Name,
    [string]$Case,
    [int]$Requests,
    [string]$Status,
    [int]$ExitCode,
    [string]$Path,
    [string[]]$Reports
) {
    [void]$sessions.Add([ordered]@{
        name = $Name
        kind = "ibs"
        case = $Case
        requests = $Requests
        status = $Status
        exit_code = $ExitCode
        path = $Path
        reports = $Reports
    })
    Save-Manifest
}

Save-Manifest
Invoke-External $UProfCli @("--version") `
    (Join-Path $CompletionRoot "uprof-version.log") | Out-Null
Invoke-External $UProfCli @("info", "--system") `
    (Join-Path $CompletionRoot "uprof-system.log") | Out-Null
Invoke-External $UProfCli @("info", "--collect-config", "ibs") `
    (Join-Path $CompletionRoot "uprof-ibs-config.log") | Out-Null

$reportViews = @(
    [pscustomobject]@{ Name = "all-op"; View = ""; Sort = "ibs-op"; Percentage = $true },
    [pscustomobject]@{ Name = "all-fetch"; View = ""; Sort = "ibs-fetch"; Percentage = $true },
    [pscustomobject]@{ Name = "op-overview"; View = "ibs_op_overview"; Sort = "ibs-op"; Percentage = $false },
    [pscustomobject]@{ Name = "memory-overview"; View = "ibs_op_ls_overview"; Sort = "ibs-op"; Percentage = $false },
    [pscustomobject]@{ Name = "load-source"; View = "ibs_op_ld"; Sort = "ibs-op"; Percentage = $false },
    [pscustomobject]@{ Name = "load-latency"; View = "ibs_op_ld_lat"; Sort = "ibs-op"; Percentage = $false },
    [pscustomobject]@{ Name = "fetch-overview"; View = "ibs_fetch_overall"; Sort = "ibs-fetch"; Percentage = $false },
    [pscustomobject]@{ Name = "fetch-itlb"; View = "ibs_fetch_itlb"; Sort = "ibs-fetch"; Percentage = $false }
)

function Get-RawIbsBytes([string]$Session) {
    $rawFiles = @(Get-ChildItem -LiteralPath $Session -Recurse -Filter "*.prd" `
        -File -ErrorAction SilentlyContinue)
    if ($rawFiles.Count -eq 0) { return 0L }
    return [long](($rawFiles | Measure-Object -Property Length -Sum).Sum)
}

function Test-ReportHasPluginSamples([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }

    # uProf emits several metadata lines containing the plugin path. A real
    # function row has a numeric event count in its second CSV field.
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^(?:"(?:[^"]|"")*"|[^,]+),"?[0-9]' -and
            $line -match 'dsmvc\.dll"?$') {
            return $true
        }
    }
    return $false
}

function Test-ReportSet([string]$Session) {
    # An empty uProf IBS container is 968 bytes on this runtime. Require room
    # for actual samples before accepting or reusing a session.
    if ((Get-RawIbsBytes $Session) -le 4096) { return $false }
    foreach ($view in $reportViews) {
        $path = Join-Path $Session "report-$($view.Name).csv"
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
        if ((Get-Item -LiteralPath $path).Length -eq 0) { return $false }
    }
    $loadSource = Join-Path $Session "report-load-source.csv"
    return Test-ReportHasPluginSamples $loadSource
}

function Invoke-IbsProfile(
    [string]$Name,
    [string]$Case,
    [int]$Frames,
    [int]$Requests,
    [string]$Script
) {
    $baseSession = Join-Path $CompletionRoot $Name
    $candidates = @($baseSession)
    $candidates += @(Get-ChildItem -LiteralPath $CompletionRoot -Directory `
        -Filter "$Name-retry*" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName)
    $completed = $candidates | Where-Object { Test-ReportSet $_ } |
        Select-Object -First 1
    if ($completed) {
        $reports = @($reportViews | ForEach-Object {
            Join-Path $completed "report-$($_.Name).csv"
        })
        Write-RunLog "Skipping complete IBS session $(Split-Path $completed -Leaf)"
        Add-Session (Split-Path $completed -Leaf) $Case $Requests `
            "existing" 0 $completed $reports
        return
    }

    $session = $baseSession
    if (Test-Path -LiteralPath $session) {
        $attempt = 1
        do {
            $session = "$baseSession-retry$attempt"
            ++$attempt
        } while (Test-Path -LiteralPath $session)
    }
    $sessionName = Split-Path $session -Leaf
    $vspipeArgs = New-VSPipeArguments $Case $Frames $Requests $Script
    $collectArgs = @(
        "collect", "--config", "ibs", "-a",
        "-o", $session, $VSPipe
    ) + $vspipeArgs
    $collect = Invoke-External $UProfCli $collectArgs `
        (Join-Path $CompletionRoot "$sessionName-collect.log") -AllowFailure
    if ($collect.ExitCode -ne 0) {
        Add-Session $sessionName $Case $Requests "collect_failed" `
            $collect.ExitCode $session @()
        $script:StopAfterFailure = $true
        return
    }
    $rawBytes = Get-RawIbsBytes $session
    if ($rawBytes -le 4096) {
        Write-RunLog "$sessionName produced only $rawBytes bytes of IBS raw data."
        Add-Session $sessionName $Case $Requests "raw_data_empty" 4 `
            $session @()
        $script:StopAfterFailure = $true
        return
    }

    $reports = [Collections.Generic.List[string]]::new()
    $failed = $false
    $failureCode = 0
    foreach ($view in $reportViews) {
        $destination = Join-Path $session "report-$($view.Name).csv"
        $reportArgs = @(
            "report", "-i", $session, "--detail", "--inline",
            "--cutoff", "200", "--symbol-path", $BinaryDir,
            "--bin-path", $BinaryDir, "--src-path", $RepoRoot,
            "--stdout", "-s", "event=$($view.Sort)"
        )
        if ($view.View) { $reportArgs += @("--view", $view.View) }
        if ($view.Percentage) { $reportArgs += "--show-percentage" }
        $report = Invoke-External $UProfCli $reportArgs `
            (Join-Path $CompletionRoot `
                "$sessionName-report-$($view.Name).log") -AllowFailure
        if ($report.ExitCode -eq 0 -and $report.Output.Count -gt 0) {
            $report.Output | Set-Content -LiteralPath $destination -Encoding UTF8
            $reports.Add($destination)
        } else {
            $failed = $true
            $failureCode = [Math]::Max($failureCode, $report.ExitCode)
        }
    }
    if (-not $failed) {
        $loadSource = Join-Path $session "report-load-source.csv"
        if (-not (Test-ReportHasPluginSamples $loadSource)) {
            Write-RunLog "$sessionName produced no dsmvc IBS OP data."
            $failed = $true
            $failureCode = 3
        }
    }
    $status = if ($failed) { "report_failed" } else { "complete" }
    Add-Session $sessionName $Case $Requests $status $failureCode `
        $session $reports.ToArray()
    if ($failed) {
        $script:StopAfterFailure = $true
    }
}

$jobs = @(
    [pscustomobject]@{ Name = "ibs-bilinear_b1-r32"; Case = "bilinear_b1"; Frames = $KernelFrames; Requests = 32; Script = $KernelScript },
    [pscustomobject]@{ Name = "ibs-bicubic_b3-r1"; Case = "bicubic_b3"; Frames = $KernelFrames; Requests = 1; Script = $KernelScript },
    [pscustomobject]@{ Name = "ibs-bicubic_b3-r8"; Case = "bicubic_b3"; Frames = $KernelFrames; Requests = 8; Script = $KernelScript },
    [pscustomobject]@{ Name = "ibs-bicubic_b3-r32"; Case = "bicubic_b3"; Frames = $KernelFrames; Requests = 32; Script = $KernelScript },
    [pscustomobject]@{ Name = "ibs-lanczos3_b5-r32"; Case = "lanczos3_b5"; Frames = $KernelFrames; Requests = 32; Script = $KernelScript },
    [pscustomobject]@{ Name = "ibs-spline36_b5-r32"; Case = "spline36_b5"; Frames = $KernelFrames; Requests = 32; Script = $KernelScript },
    [pscustomobject]@{ Name = "ibs-spline64_b7-r32"; Case = "spline64_b7"; Frames = $KernelFrames; Requests = 32; Script = $KernelScript },
    [pscustomobject]@{ Name = "ibs-sweep-getfnative-r32"; Case = "getfnative"; Frames = $SweepFrames; Requests = 32; Script = $SweepScript },
    [pscustomobject]@{ Name = "ibs-sweep-getfnative_v2-r32"; Case = "getfnative_v2"; Frames = $SweepFrames; Requests = 32; Script = $SweepScript },
    [pscustomobject]@{ Name = "ibs-sweep-selectkernel-r32"; Case = "selectkernel"; Frames = $SweepFrames; Requests = 32; Script = $SweepScript }
)

$script:StopAfterFailure = $false
foreach ($job in $jobs) {
    Invoke-IbsProfile $job.Name $job.Case $job.Frames `
        $job.Requests $job.Script
    if ($script:StopAfterFailure) {
        Write-RunLog "Stopping the IBS pass after the first invalid session."
        break
    }
}

$manifest.completed_utc = [DateTimeOffset]::UtcNow.ToString("o")
Save-Manifest
$failedSessions = @($sessions | Where-Object {
    $_.status -notin @("complete", "existing")
})
if ($failedSessions.Count -gt 0) {
    Write-RunLog "$($failedSessions.Count) IBS sessions did not complete."
    exit 2
}
Write-RunLog "All IBS completion sessions finished successfully."
exit 0
