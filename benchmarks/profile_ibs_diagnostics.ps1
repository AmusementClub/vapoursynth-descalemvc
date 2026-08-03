[CmdletBinding()]
param(
    [string]$Output = "",
    [int]$Threads = 32,
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
$VSPipe = Join-Path $RuntimeDir "VSPipe.exe"
$KernelScript = Join-Path $SnapshotDir "vspipe_benchmark.vpy"
$Image = "C:\Users\lsy39\Downloads\6.2-1.png"
$OldPlugin = `
    "D:\okegui\OKEGui\tools\vapoursynth\vapoursynth64\plugins\descale.dll"
$UProfCli = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Escape-SingleQuoted([string]$Value) {
    return $Value.Replace("'", "''")
}

foreach ($path in @($Plugin, $VSPipe, $KernelScript, $Image, $OldPlugin,
        $UProfCli)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "required file does not exist: $path"
    }
}
New-Item -ItemType Directory -Path $CompletionRoot -Force | Out-Null

if (-not $Elevated) {
    $hostExe = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    $worker = @(
        "& '$(Escape-SingleQuoted $PSCommandPath)'",
        "-Elevated",
        "-Output '$(Escape-SingleQuoted $Output)'",
        "-Threads $Threads"
    ) -join " "
    $bootstrapLog = Join-Path $CompletionRoot "ibs-diagnostics-bootstrap.log"
    $invoke = @"
`$ErrorActionPreference = 'Stop'
try {
    $worker *>&1 | ForEach-Object {
        `$_ | Out-File -LiteralPath '$(Escape-SingleQuoted $bootstrapLog)' -Append -Encoding utf8
        Write-Host `$_
    }
    if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
    exit 0
} catch {
    (`$_ | Format-List * -Force | Out-String) | Out-File -LiteralPath '$(Escape-SingleQuoted $bootstrapLog)' -Append -Encoding utf8
    exit 1
}
"@
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($invoke))
    Write-Host "Requesting one elevated worker for all IBS diagnostics."
    $process = Start-Process -FilePath $hostExe -Verb RunAs `
        -WindowStyle Hidden -ArgumentList `
        "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encoded" `
        -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "elevated IBS diagnostics failed; see $bootstrapLog"
    }
    exit 0
}

if (-not (Test-Administrator)) {
    throw "IBS diagnostics are not running as administrator"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$DiagnosticRoot = Join-Path $CompletionRoot "ibs-diagnostics-$stamp"
New-Item -ItemType Directory -Path $DiagnosticRoot -Force | Out-Null
$RunLog = Join-Path $DiagnosticRoot "diagnostics.log"
$results = [Collections.ArrayList]::new()

function Write-DiagnosticLog([string]$Message) {
    $line = "{0:o} {1}" -f [DateTimeOffset]::Now, $Message
    $line | Tee-Object -FilePath $RunLog -Append | Write-Host
}

function Invoke-External(
    [string]$Executable,
    [string[]]$Arguments,
    [string]$LogPath
) {
    Write-DiagnosticLog "& '$Executable' $($Arguments -join ' ')"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $lines = & $Executable @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    $lines | Set-Content -LiteralPath $LogPath -Encoding UTF8
    return [pscustomobject]@{ ExitCode = $exitCode; Output = @($lines) }
}

function New-VSPipeArguments([int]$Frames) {
    return @(
        "--arg", "implementation=new",
        "--arg", "plugin=$Plugin",
        "--arg", "old_plugin=$OldPlugin",
        "--arg", "image=$Image",
        "--arg", "case=bilinear_b1",
        "--arg", "frames=$Frames",
        "--arg", "threads=$Threads",
        "--requests", "32",
        "--end", "$($Frames - 1)",
        "--filter-time", $KernelScript, "."
    )
}

function Get-RawBytes([string]$Session) {
    $files = @(Get-ChildItem -LiteralPath $Session -Recurse -Filter "*.prd" `
        -File -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return 0L }
    return [long](($files | Measure-Object -Property Length -Sum).Sum)
}

function Add-Result(
    [string]$Name,
    [string]$Mode,
    [int]$ExitCode,
    [string]$Session,
    [long]$RawBytes,
    [string]$Report
) {
    $valid = $ExitCode -eq 0 -and $RawBytes -gt 4096
    [void]$results.Add([ordered]@{
        name = $Name
        mode = $Mode
        exit_code = $ExitCode
        raw_bytes = $RawBytes
        valid_raw_data = $valid
        session = $Session
        report = $Report
    })
    Write-DiagnosticLog "$Name exit=$ExitCode raw_bytes=$RawBytes valid=$valid"
}

function Export-ProbeReport(
    [string]$Name,
    [string]$Session,
    [string]$Event
) {
    if ((Get-RawBytes $Session) -le 4096) { return "" }
    $path = Join-Path $DiagnosticRoot "$Name-report.csv"
    $report = Invoke-External $UProfCli @(
        "report", "-i", $Session, "--detail", "--inline",
        "--cutoff", "200", "--symbol-path", $BinaryDir,
        "--bin-path", $BinaryDir, "--stdout", "-s", "event=$Event"
    ) (Join-Path $DiagnosticRoot "$Name-report.log")
    if ($report.ExitCode -eq 0) {
        $report.Output | Set-Content -LiteralPath $path -Encoding UTF8
        return $path
    }
    return ""
}

function Invoke-LaunchProbe(
    [string]$Name,
    [string[]]$ProfileArguments,
    [string]$ReportEvent
) {
    $session = Join-Path $DiagnosticRoot $Name
    $debugDir = Join-Path $DiagnosticRoot "$Name-debug"
    New-Item -ItemType Directory -Path $debugDir -Force | Out-Null
    $args = @("collect") + $ProfileArguments + @(
        "--log-level", "5", "--enable-logts", "--log-path", $debugDir,
        "-o", $session, $VSPipe
    ) + (New-VSPipeArguments 4096)
    $collect = Invoke-External $UProfCli $args `
        (Join-Path $DiagnosticRoot "$Name-collect.log")
    $rawBytes = Get-RawBytes $session
    $report = Export-ProbeReport $Name $session $ReportEvent
    Add-Result $Name "launch" $collect.ExitCode $session $rawBytes $report
}

function Invoke-DetachedProbe {
    $name = "config-detached-swp"
    $session = Join-Path $DiagnosticRoot $name
    $debugDir = Join-Path $DiagnosticRoot "$name-debug"
    New-Item -ItemType Directory -Path $debugDir -Force | Out-Null
    $stdout = Join-Path $DiagnosticRoot "$name-vspipe.stdout.log"
    $stderr = Join-Path $DiagnosticRoot "$name-vspipe.stderr.log"
    $vspipeProcess = Start-Process -FilePath $VSPipe -WindowStyle Hidden `
        -WorkingDirectory $RepoRoot -ArgumentList (New-VSPipeArguments 16384) `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Start-Sleep -Milliseconds 750
    try {
        $collect = Invoke-External $UProfCli @(
            "collect", "--config", "ibs", "-a", "-d", "5",
            "--log-level", "5", "--enable-logts", "--log-path", $debugDir,
            "-o", $session
        ) (Join-Path $DiagnosticRoot "$name-collect.log")
    } finally {
        if (-not $vspipeProcess.HasExited) {
            $vspipeProcess.WaitForExit()
        }
    }
    $rawBytes = Get-RawBytes $session
    $report = Export-ProbeReport $name $session "ibs-op"
    Add-Result $name "detached" $collect.ExitCode $session $rawBytes $report
}

Write-DiagnosticLog "Starting IBS diagnostic matrix as administrator."
Invoke-LaunchProbe "explicit-both-swp" @(
    "-e", "event=ibs-op,interval=50000",
    "-e", "event=ibs-fetch,interval=50000", "-a"
) "ibs-op"
Invoke-LaunchProbe "explicit-op-swp" @(
    "-e", "event=ibs-op,interval=50000", "-a"
) "ibs-op"
Invoke-LaunchProbe "explicit-fetch-swp" @(
    "-e", "event=ibs-fetch,interval=50000", "-a"
) "ibs-fetch"
Invoke-DetachedProbe

if (@($results | Where-Object valid_raw_data).Count -eq 0) {
    $busy = @(Get-Process AMDuProfCLI,AMDuProf,VSPipe `
        -ErrorAction SilentlyContinue)
    if ($busy.Count -eq 0) {
        Write-DiagnosticLog "All probes were empty; restarting AMDCpuProfiler."
        $serviceLog = Join-Path $DiagnosticRoot "driver-restart.log"
        $stopOutput = & sc.exe stop AMDCpuProfiler 2>&1
        $stopOutput | Set-Content -LiteralPath $serviceLog -Encoding UTF8
        Start-Sleep -Seconds 1
        $startOutput = & sc.exe start AMDCpuProfiler 2>&1
        $startOutput | Add-Content -LiteralPath $serviceLog -Encoding UTF8
        Start-Sleep -Seconds 1
        Invoke-LaunchProbe "config-after-driver-restart" @(
            "--config", "ibs", "-a"
        ) "ibs-op"
    } else {
        Write-DiagnosticLog "Skipping driver restart because profiler processes are active."
    }
}

$manifest = [ordered]@{
    schema_version = 1
    created_utc = [DateTimeOffset]::UtcNow.ToString("o")
    elevated = $true
    uprof_version = (& $UProfCli --version 2>&1) -join "`n"
    cpu_driver = (Get-Item `
        "C:\Windows\System32\drivers\AMDCpuProfiler.sys").VersionInfo.ProductVersion
    operating_system = Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption, Version, BuildNumber
    results = $results
}
$manifest | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $DiagnosticRoot "manifest.json") `
        -Encoding UTF8
Write-DiagnosticLog "IBS diagnostic matrix finished."
$results | Format-Table name, mode, exit_code, raw_bytes, valid_raw_data
exit 0
