[CmdletBinding()]
param(
    [string]$Output = "",
    [int]$Threads = 32,
    [int]$KernelFrames = 8192,
    [int]$PcmFrames = 16384,
    [int]$SweepFrames = 500,
    [switch]$SkipBuild,
    [switch]$SkipEtw,
    [switch]$PrepareOnly,
    [switch]$TbpOnly,
    [switch]$PcmOnly,
    [switch]$Elevated
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VsRoot = "D:\okegui\OKEGui\tools\vapoursynth"
$Image = "C:\Users\lsy39\Downloads\6.2-1.png"
$UProfCli = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"
$UProfPcm = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfPcm.exe"
$Wpr = Join-Path $env:SystemRoot "System32\wpr.exe"
$xperfCommand = Get-Command xperf.exe -ErrorAction SilentlyContinue
$Xperf = if ($xperfCommand) {
    $xperfCommand.Source
} else {
    Join-Path ([Environment]::GetFolderPath("ProgramFilesX86")) `
        "Windows Kits\10\Windows Performance Toolkit\xperf.exe"
}
$ExpectedImageHash = "61F9EE1AC858BBADD6A959BA35F5ECEB077B8452B91E97A5CE3D39EBC69E20C6"

function Get-GitText([string[]]$Arguments) {
    $value = & git -C $RepoRoot @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git failed: $($value -join [Environment]::NewLine)"
    }
    return ($value -join [Environment]::NewLine).Trim()
}

$Commit = Get-GitText @("rev-parse", "HEAD")
$ShortCommit = $Commit.Substring(0, 8)
if (-not $Output) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Output = Join-Path $RepoRoot "benchmark-results\profile-current-$ShortCommit-$stamp"
}
$Output = [IO.Path]::GetFullPath($Output)

function Escape-SingleQuoted([string]$Value) {
    return $Value.Replace("'", "''")
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

New-Item -ItemType Directory -Path $Output -Force | Out-Null
$MasterLog = Join-Path $Output "profile-run.log"
$CommandLog = Join-Path $Output "commands.ps1"

function Write-RunLog([string]$Message) {
    $line = "{0:o} {1}" -f [DateTimeOffset]::Now, $Message
    $line | Tee-Object -FilePath $MasterLog -Append | Write-Host
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

function Get-CMakePath {
    $cache = Join-Path $RepoRoot "build\CMakeCache.txt"
    if (Test-Path -LiteralPath $cache) {
        $line = Get-Content -LiteralPath $cache |
            Where-Object { $_.StartsWith("CMAKE_COMMAND:INTERNAL=") } |
            Select-Object -First 1
        if ($line) {
            $candidate = $line.Substring($line.IndexOf("=") + 1)
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
    }
    $command = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "CMake was not found."
}

if (-not $SkipBuild) {
    $cmake = Get-CMakePath
    Invoke-External $cmake @(
        "--build", (Join-Path $RepoRoot "build"),
        "--config", "RelWithDebInfo",
        "--target", "dsmvc", "dsmvc_engine_tests"
    ) (Join-Path $Output "build.log") | Out-Null
    $ctest = Join-Path (Split-Path $cmake -Parent) "ctest.exe"
    Invoke-External $ctest @(
        "--test-dir", (Join-Path $RepoRoot "build"),
        "-C", "RelWithDebInfo", "--output-on-failure"
    ) (Join-Path $Output "tests.log") | Out-Null
}

foreach ($required in @($UProfCli, $UProfPcm, $Image,
        (Join-Path $VsRoot "VSPipe.exe"))) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path does not exist: $required"
    }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Image).Hash -ne $ExpectedImageHash) {
    throw "The profile input hash does not match 6.2-1.png."
}

$BuildDir = Join-Path $RepoRoot "build\RelWithDebInfo"

# A fixed output directory is resumable only while it still identifies the
# same binary.  Never mix sessions from an older DLL with a new executable:
# redirect to a sibling output and keep the old evidence intact instead of
# throwing from the elevated worker (which made the UAC console disappear).
$sourceHashes = @{}
foreach ($name in @("dsmvc.dll", "dsmvc.pdb")) {
    $source = Join-Path $BuildDir $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing build output: $source"
    }
    $sourceHashes[$name] = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash
}
$existingBinaryDir = Join-Path $Output "binary"
$binaryMismatch = [Collections.Generic.List[string]]::new()
foreach ($name in $sourceHashes.Keys) {
    $target = Join-Path $existingBinaryDir $name
    if (Test-Path -LiteralPath $target) {
        $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
        if ($targetHash -ne $sourceHashes[$name]) {
            [void]$binaryMismatch.Add($name)
        }
    }
}
if ($binaryMismatch.Count -gt 0) {
    $oldOutput = $Output
    $refreshStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Output = "$Output-binary-refresh-$refreshStamp"
    New-Item -ItemType Directory -Path $Output -Force | Out-Null
    $MasterLog = Join-Path $Output "profile-run.log"
    $CommandLog = Join-Path $Output "commands.ps1"
    @(
        "Profile output was redirected because the requested output contained a different binary.",
        "old_output=$oldOutput",
        "new_output=$Output",
        "mismatched_files=$($binaryMismatch -join ',')"
    ) | Set-Content -LiteralPath (Join-Path $Output "binary-refresh.log") -Encoding UTF8
    Write-RunLog "Existing output has different $($binaryMismatch -join ', '); using fresh output $Output"
}
$BinaryDir = Join-Path $Output "binary"
New-Item -ItemType Directory -Path $BinaryDir -Force | Out-Null
foreach ($name in @("dsmvc.dll", "dsmvc.pdb")) {
    $source = Join-Path $BuildDir $name
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing build output: $source" }
    $target = Join-Path $BinaryDir $name
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash -ne
            (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash) {
            throw "Profile output binary refresh failed for ${name}: $target"
        }
    } else {
        Copy-Item -LiteralPath $source -Destination $target
    }
}
$Plugin = Join-Path $BinaryDir "dsmvc.dll"

$SnapshotDir = Join-Path $Output "snapshot"
New-Item -ItemType Directory -Path $SnapshotDir -Force | Out-Null
Get-GitText @("status", "--short") |
    Set-Content -LiteralPath (Join-Path $SnapshotDir "git-status.txt") -Encoding UTF8
Get-GitText @("diff", "--", "CMakeLists.txt", "include", "src") |
    Set-Content -LiteralPath (Join-Path $SnapshotDir "engine-source.diff") -Encoding UTF8
if (-not (Test-Path -LiteralPath (Join-Path $SnapshotDir "source-head.zip"))) {
    Invoke-External "git" @(
        "-C", $RepoRoot, "archive", "--format=zip",
        "--output=$(Join-Path $SnapshotDir 'source-head.zip')", "HEAD"
    ) (Join-Path $SnapshotDir "git-archive.log") | Out-Null
}
Copy-Item -LiteralPath (Join-Path $RepoRoot "benchmarks\vspipe_benchmark.vpy") `
    -Destination (Join-Path $SnapshotDir "vspipe_benchmark.vpy") -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "benchmarks\vspipe_getnative.vpy") `
    -Destination (Join-Path $SnapshotDir "vspipe_getnative.vpy") -Force
Copy-Item -LiteralPath $PSCommandPath `
    -Destination (Join-Path $SnapshotDir "profile_current_cpu.ps1") -Force
$summaryScript = Join-Path $RepoRoot "benchmarks\summarize_cpu_profile.py"
if (Test-Path -LiteralPath $summaryScript) {
    Copy-Item -LiteralPath $summaryScript `
        -Destination (Join-Path $SnapshotDir "summarize_cpu_profile.py") -Force
}

function New-IsolatedRuntime {
    param([string]$Destination)
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $rootFiles = @(
        "VSPipe.exe", "VSScript.dll", "VapourSynth.dll", "portable.vs",
        "python3.dll", "python311.dll", "python311.zip", "python311.pth",
        "python311._pth", "msvcp140.dll", "vcruntime140.dll",
        "vcruntime140_1.dll", "mimalloc-override.dll", "mimalloc-redirect.dll"
    )
    foreach ($name in $rootFiles) {
        $source = Join-Path $VsRoot $name
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $Destination $name) -Force
        }
    }
    foreach ($name in @("DLLs", "Lib", "VapourSynthScripts")) {
        $link = Join-Path $Destination $name
        if (-not (Test-Path -LiteralPath $link)) {
            New-Item -ItemType Junction -Path $link -Target (Join-Path $VsRoot $name) | Out-Null
        }
    }
    $corePlugins = Join-Path $Destination "vapoursynth64\coreplugins"
    $plugins = Join-Path $Destination "vapoursynth64\plugins"
    New-Item -ItemType Directory -Path $corePlugins -Force | Out-Null
    New-Item -ItemType Directory -Path $plugins -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $VsRoot "vapoursynth64\coreplugins\Imwri.dll") `
        -Destination (Join-Path $corePlugins "Imwri.dll") -Force
}

$RuntimeDir = Join-Path $Output "isolated-vs"
New-IsolatedRuntime $RuntimeDir
$VSPipe = Join-Path $RuntimeDir "VSPipe.exe"
$KernelScript = Join-Path $SnapshotDir "vspipe_benchmark.vpy"
$SweepScript = Join-Path $SnapshotDir "vspipe_getnative.vpy"

function New-VSPipeArguments(
    [string]$Case,
    [int]$Frames,
    [int]$Requests,
    [string]$Script
) {
    return @(
        "--arg", "implementation=new",
        "--arg", "plugin=$Plugin",
        "--arg", "old_plugin=$(Join-Path $VsRoot 'vapoursynth64\plugins\descale.dll')",
        "--arg", "image=$Image",
        "--arg", "case=$Case",
        "--arg", "frames=$Frames",
        "--arg", "threads=$Threads",
        "--requests", "$Requests",
        "--end", "$($Frames - 1)",
        "--filter-time", $Script, "."
    )
}

$smokeArgs = New-VSPipeArguments "bicubic_b3" 2 1 $KernelScript
Invoke-External $VSPipe $smokeArgs (Join-Path $Output "isolated-runtime-smoke.log") | Out-Null
if ($PrepareOnly) {
    Write-RunLog "Preparation and isolated-runtime smoke test completed."
    exit 0
}

# Build, snapshot, and smoke-test without elevation. The elevated pass is then
# forced to reuse the exact prepared binary so profiling identity cannot drift.
if (-not $Elevated -and -not $PrepareOnly) {
    $hostExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    $workerArguments = @(
        "& '$(Escape-SingleQuoted $PSCommandPath)'",
        "-Elevated",
        "-SkipBuild",
        "-Output '$(Escape-SingleQuoted $Output)'",
        "-Threads $Threads",
        "-KernelFrames $KernelFrames",
        "-PcmFrames $PcmFrames",
        "-SweepFrames $SweepFrames"
    )
    if ($SkipEtw) { $workerArguments += "-SkipEtw" }
    if ($TbpOnly) { $workerArguments += "-TbpOnly" }
    if ($PcmOnly) { $workerArguments += "-PcmOnly" }
    $bootstrapLog = "$Output.bootstrap.log"
    "Waiting for elevated worker." |
        Set-Content -LiteralPath $bootstrapLog -Encoding UTF8
    $worker = $workerArguments -join " "
    $invoke = @"
`$ErrorActionPreference = 'Stop'
try {
    $worker *>&1 | ForEach-Object {
        `$_ | Out-File -LiteralPath '$(Escape-SingleQuoted $bootstrapLog)' -Append -Encoding utf8
        Write-Host `$_
    }
    exit 0
} catch {
    (`$_ | Format-List * -Force | Out-String) | Out-File -LiteralPath '$(Escape-SingleQuoted $bootstrapLog)' -Append -Encoding utf8
    exit 1
}
"@
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($invoke))
    Write-Host "Preparation passed. Requesting one elevated PowerShell session for all AMD PMU operations."
    $process = Start-Process -FilePath $hostExe -Verb RunAs `
        -ArgumentList "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded" `
        -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        Write-Error "Elevated worker failed; see $bootstrapLog"
    }
    exit $process.ExitCode
}

if (-not (Test-Administrator)) {
    throw "The elevated profiling worker is not running as administrator."
}

$cliVersionResult = Invoke-External $UProfCli @("--version") `
    (Join-Path $Output "uprof-cli-version.log")
$pcmVersionResult = Invoke-External $UProfPcm @("-v") `
    (Join-Path $Output "uprof-pcm-version.log")
$assessConfigPath = Join-Path $Output "uprof-assess-ext-config.log"
Invoke-External $UProfCli @("info", "--collect-config", "assess_ext") `
    $assessConfigPath | Out-Null

$ManifestPath = Join-Path $Output "manifest.json"
$preservedSessions = [Collections.ArrayList]::new()
$previousManifest = $null
if (($TbpOnly -or $PcmOnly) -and (Test-Path -LiteralPath $ManifestPath)) {
    $previousManifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($entry in $previousManifest.sessions) {
        if ($TbpOnly -and $entry.kind -eq "tbp") { continue }
        if ($PcmOnly -and $entry.kind -eq "pcm") { continue }
        $status = $entry.status
        if ($entry.kind -eq "process_memory" -and [int]$entry.exit_code -eq 0) {
            $status = "complete"
        }
        [void]$preservedSessions.Add([ordered]@{
            name = $entry.name
            kind = $entry.kind
            case = $entry.case
            requests = [int]$entry.requests
            status = $status
            exit_code = [int]$entry.exit_code
            path = $entry.path
        })
    }
}

$passStartedUtc = [DateTimeOffset]::UtcNow.ToString("o")
$passMode = $(if ($TbpOnly) { "tbp_only" } elseif ($PcmOnly) { "pcm_only" } else { "full" })
$passElevated = Test-Administrator
$manifest = [ordered]@{
    schema_version = 1
    started_utc = $(if ($previousManifest) { $previousManifest.started_utc } else { $passStartedUtc })
    completed_utc = $null
    git_commit = $Commit
    git_status = Get-GitText @("status", "--short")
    elevated = $(if ($previousManifest) { [bool]$previousManifest.elevated } else { $passElevated })
    mode = $(if ($previousManifest) { $previousManifest.mode } else { $passMode })
    last_pass_started_utc = $passStartedUtc
    last_pass_completed_utc = $null
    last_pass_elevated = $passElevated
    last_pass_mode = $passMode
    cpu = (Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores,
        NumberOfLogicalProcessors, MaxClockSpeed)
    operating_system = (Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption, Version, BuildNumber, TotalVisibleMemorySize)
    threads = $Threads
    kernel_frames = $KernelFrames
    pcm_frames = $PcmFrames
    sweep_frames = $SweepFrames
    image = $Image
    image_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Image).Hash
    plugin = $Plugin
    plugin_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Plugin).Hash
    pdb_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $BinaryDir "dsmvc.pdb")).Hash
    vspipe = $VSPipe
    vspipe_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $VSPipe).Hash
    uprof_cli_version = (($cliVersionResult.Output | ForEach-Object { "$_" }) -join " ").Trim()
    uprof_pcm_version = (($pcmVersionResult.Output | ForEach-Object { "$_" }) -join " ").Trim()
    profile_backend = "AMD uProf hardware PMU (assess_ext) + AMD uProf PCM memory"
    hardware_pmu_config = "assess_ext"
    hardware_pmu_config_log = $assessConfigPath
    iba_used = $false
    skip_etw = [bool]$SkipEtw
    isolated_plugins = @("Imwri.dll", "dsmvc.dll (explicit LoadPlugin)")
    sessions = $preservedSessions
}

function Save-Manifest {
    $manifest | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}

function Complete-Manifest {
    $completedUtc = [DateTimeOffset]::UtcNow.ToString("o")
    $manifest.completed_utc = $completedUtc
    $manifest.last_pass_completed_utc = $completedUtc
    Save-Manifest
}

function Add-Session(
    [string]$Name,
    [string]$Kind,
    [string]$Case,
    [int]$Requests,
    [string]$Status,
    [int]$ExitCode,
    [string]$Path
) {
    [void]$manifest.sessions.Add([ordered]@{
        name = $Name
        kind = $Kind
        case = $Case
        requests = $Requests
        status = $Status
        exit_code = $ExitCode
        path = $Path
    })
    Save-Manifest
}

Save-Manifest

function Invoke-CpuProfile(
    [string]$Name,
    [string]$Config,
    [string]$Case,
    [int]$Frames,
    [int]$Requests,
    [string]$Script,
    [switch]$CallGraph
) {
    $baseSession = Join-Path $Output $Name
    $candidates = @($baseSession)
    $candidates += @(Get-ChildItem -LiteralPath $Output -Directory `
        -Filter "$Name-retry*" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName)
    $completed = $candidates | Where-Object {
        Test-Path -LiteralPath (Join-Path $_ "report-percentage.csv")
    } | Select-Object -First 1
    if ($completed) {
        $completedName = Split-Path $completed -Leaf
        Write-RunLog "Skipping completed session $completedName"
        Add-Session $completedName $Config $Case $Requests "existing" 0 $completed
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
    $report = Join-Path $session "report-percentage.csv"
    $vspipeArgs = New-VSPipeArguments $Case $Frames $Requests $Script
    $collectArgs = @("collect", "--config", $Config)
    if ($CallGraph) {
        $collectArgs += @(
            "--call-graph-mode", "fpo", "--call-graph-type", "user")
    }
    $collectArgs += @("-o", $session, $VSPipe) + $vspipeArgs
    $collectResult = Invoke-External $UProfCli $collectArgs `
        (Join-Path $Output "$sessionName-collect.log") -AllowFailure
    if ($collectResult.ExitCode -ne 0) {
        Add-Session $sessionName $Config $Case $Requests "collect_failed" `
            $collectResult.ExitCode $session
        return
    }
    $reportArgs = @(
        "report", "-i", $session, "--detail", "--inline",
        "--show-percentage", "--cutoff", "200",
        "--symbol-path", $BinaryDir, "--bin-path", $BinaryDir,
        "--src-path", $RepoRoot, "--stdout")
    if ($CallGraph) { $reportArgs += "-g" }
    $reportResult = Invoke-External $UProfCli $reportArgs `
        (Join-Path $Output "$sessionName-report.log") -AllowFailure
    if ($reportResult.ExitCode -eq 0) {
        $reportResult.Output | Set-Content -LiteralPath $report -Encoding UTF8
        Add-Session $sessionName $Config $Case $Requests "complete" 0 $session
    } else {
        Add-Session $sessionName $Config $Case $Requests "report_failed" `
            $reportResult.ExitCode $session
    }
}

function Invoke-PcmProfile(
    [string]$Name,
    [string]$Case,
    [int]$Requests
) {
    $baseSession = Join-Path $Output $Name
    $candidates = @($baseSession)
    $candidates += @(Get-ChildItem -LiteralPath $Output -Directory `
        -Filter "$Name-retry*" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName)
    $completed = $null
    foreach ($candidate in $candidates) {
        $cumulative = Get-ChildItem -LiteralPath $candidate -Recurse `
            -Filter "report-cumulative.csv" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $cumulative) { continue }
        $bandwidth = Select-String -LiteralPath $cumulative.FullName `
            -Pattern '^Total Mem Bw \(GB/s\),(?<value>[0-9.]+)' |
            Select-Object -Last 1
        if ($bandwidth -and [double]$bandwidth.Matches[0].Groups["value"].Value -gt 0) {
            $completed = $candidate
            break
        }
    }
    if ($completed) {
        $completedName = Split-Path $completed -Leaf
        Write-RunLog "Skipping completed PCM session $completedName"
        Add-Session $completedName "pcm" $Case $Requests "existing" 0 $completed
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
    $vspipeArgs = New-VSPipeArguments $Case $PcmFrames $Requests $KernelScript
    $args = @(
        "profile", "-m", "memory", "-a", "-d", "10",
        "-I", "1000", "--start-delay", "1000", "-O", $session,
        "--", $VSPipe) + $vspipeArgs
    $result = Invoke-External $UProfPcm $args `
        (Join-Path $Output "$sessionName-collect.log") -AllowFailure
    $cumulative = Get-ChildItem -LiteralPath $session -Recurse `
        -Filter "report-cumulative.csv" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $hasData = $false
    if ($cumulative) {
        $bandwidth = Select-String -LiteralPath $cumulative.FullName `
            -Pattern '^Total Mem Bw \(GB/s\),(?<value>[0-9.]+)' |
            Select-Object -Last 1
        $hasData = $bandwidth -and `
            [double]$bandwidth.Matches[0].Groups["value"].Value -gt 0
    }
    if ($result.ExitCode -eq 0 -and $hasData) {
        Add-Session $sessionName "pcm" $Case $Requests "complete" 0 $session
    } else {
        $status = if ($result.ExitCode -eq 0) { "no_data" } else { "collect_failed" }
        Add-Session $sessionName "pcm" $Case $Requests $status $result.ExitCode $session
    }
}

function Invoke-MemoryTrace(
    [string]$Name,
    [string]$Case,
    [int]$Frames,
    [int]$Requests,
    [string]$Script
) {
    $csv = Join-Path $Output "$Name.csv"
    if (Test-Path -LiteralPath $csv) {
        Write-RunLog "Skipping completed memory trace $Name"
        Add-Session $Name "process_memory" $Case $Requests "existing" 0 $csv
        return
    }
    $stdout = Join-Path $Output "$Name-vspipe.stdout.log"
    $stderr = Join-Path $Output "$Name-vspipe.stderr.log"
    $vspipeArgs = New-VSPipeArguments $Case $Frames $Requests $Script
    Add-Content -LiteralPath $CommandLog -Value (Format-Command $VSPipe $vspipeArgs) -Encoding UTF8
    Write-RunLog "Sampling process memory for $Name"
    $process = Start-Process -FilePath $VSPipe -ArgumentList $vspipeArgs `
        -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    $samples = [Collections.Generic.List[object]]::new()
    $clock = [Diagnostics.Stopwatch]::StartNew()
    while (-not $process.HasExited) {
        try {
            $process.Refresh()
            $samples.Add([pscustomobject]@{
                elapsed_ms = $clock.Elapsed.TotalMilliseconds
                working_set_bytes = $process.WorkingSet64
                peak_working_set_bytes = $process.PeakWorkingSet64
                private_bytes = $process.PrivateMemorySize64
                virtual_bytes = $process.VirtualMemorySize64
                paged_bytes = $process.PagedMemorySize64
                thread_count = $process.Threads.Count
                handle_count = $process.HandleCount
                cpu_seconds = $process.TotalProcessorTime.TotalSeconds
            })
        } catch [InvalidOperationException] {
            break
        }
        Start-Sleep -Milliseconds 50
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = [int]$process.ExitCode
    $samples | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
    $status = if ($exitCode -eq 0) { "complete" } else { "process_failed" }
    Add-Session $Name "process_memory" $Case $Requests $status $exitCode $csv
}

function Export-EtwReports([string]$Name, [string]$Etl) {
    if (-not (Test-Path -LiteralPath $Xperf)) {
        Write-RunLog "xperf is unavailable; leaving $Name as raw ETL only"
        return
    }
    $exports = @(
        [pscustomobject]@{ Suffix = "process"; Arguments = @("process") },
        [pscustomobject]@{ Suffix = "cswitch-process"; Arguments = @("cswitch", "-process") },
        [pscustomobject]@{ Suffix = "cswitch-thread"; Arguments = @("cswitch", "-thread") },
        [pscustomobject]@{ Suffix = "profile-detail"; Arguments = @("profile", "-detail") },
        [pscustomobject]@{ Suffix = "tracestats"; Arguments = @("tracestats", "-timespan", "actual") },
        [pscustomobject]@{ Suffix = "tracestats-detail"; Arguments = @("tracestats", "-detail") }
    )
    foreach ($export in $exports) {
        $destination = Join-Path $Output "$Name-$($export.Suffix).txt"
        if ((Test-Path -LiteralPath $destination) -and
            (Get-Item -LiteralPath $destination).Length -gt 0) {
            continue
        }
        $arguments = @("-i", $Etl, "-o", $destination, "-a") + $export.Arguments
        Invoke-External $Xperf $arguments `
            (Join-Path $Output "$Name-$($export.Suffix)-export.log") | Out-Null
        if (-not (Test-Path -LiteralPath $destination) -or
            (Get-Item -LiteralPath $destination).Length -eq 0) {
            throw "xperf did not create $destination"
        }
    }
}

function Invoke-EtwTrace(
    [string]$Name,
    [string]$Case,
    [int]$Frames,
    [int]$Requests,
    [string]$Script
) {
    $etl = Join-Path $Output "$Name.etl"
    if (Test-Path -LiteralPath $etl) {
        Export-EtwReports $Name $etl
        Write-RunLog "Skipping completed ETW trace $Name"
        Add-Session $Name "etw_cpu" $Case $Requests "existing" 0 $etl
        return
    }
    $start = Invoke-External $Wpr @("-start", "CPU", "-filemode") `
        (Join-Path $Output "$Name-wpr-start.log") -AllowFailure
    if ($start.ExitCode -ne 0) {
        Add-Session $Name "etw_cpu" $Case $Requests "start_failed" $start.ExitCode $etl
        return
    }
    $vspipeArgs = New-VSPipeArguments $Case $Frames $Requests $Script
    $run = Invoke-External $VSPipe $vspipeArgs `
        (Join-Path $Output "$Name-vspipe.log") -AllowFailure
    $stop = Invoke-External $Wpr @("-stop", $etl) `
        (Join-Path $Output "$Name-wpr-stop.log") -AllowFailure
    if ($run.ExitCode -eq 0 -and $stop.ExitCode -eq 0) {
        Export-EtwReports $Name $etl
        Add-Session $Name "etw_cpu" $Case $Requests "complete" 0 $etl
    } else {
        Add-Session $Name "etw_cpu" $Case $Requests "trace_failed" `
            ([Math]::Max($run.ExitCode, $stop.ExitCode)) $etl
    }
}

$kernelCases = @(
    [pscustomobject]@{ Name = "bilinear_b1"; Bandwidth = 1 },
    [pscustomobject]@{ Name = "bicubic_b3"; Bandwidth = 3 },
    [pscustomobject]@{ Name = "lanczos3_b5"; Bandwidth = 5 },
    [pscustomobject]@{ Name = "spline36_b5"; Bandwidth = 5 },
    [pscustomobject]@{ Name = "spline64_b7"; Bandwidth = 7 }
)

if (-not $PcmOnly) {
    foreach ($kernel in $kernelCases) {
        foreach ($requests in @(1, 8, 32)) {
            Invoke-CpuProfile "tbp-$($kernel.Name)-r$requests" "tbp" `
                $kernel.Name $KernelFrames $requests $KernelScript -CallGraph
        }
    }

    foreach ($case in @("getfnative", "getfnative_v2", "selectkernel")) {
        Invoke-CpuProfile "tbp-sweep-$case-r32" "tbp" `
            $case $SweepFrames 32 $SweepScript -CallGraph
    }
}

if ($TbpOnly) {
    Complete-Manifest
    Write-RunLog "TBP-only profile jobs finished. Manifest: $ManifestPath"
    exit 0
}

if (-not $PcmOnly) {
    foreach ($kernel in $kernelCases) {
        Invoke-CpuProfile "assess-$($kernel.Name)-r32" "assess_ext" `
            $kernel.Name $KernelFrames 32 $KernelScript
    }
}

foreach ($kernel in $kernelCases) {
    Invoke-PcmProfile "pcm-memory-$($kernel.Name)-r32" $kernel.Name 32
}
foreach ($requests in @(1, 8)) {
    Invoke-PcmProfile "pcm-memory-bicubic_b3-r$requests" "bicubic_b3" $requests
}

if ($PcmOnly) {
    Complete-Manifest
    Write-RunLog "PCM-only profile jobs finished. Manifest: $ManifestPath"
    exit 0
}

foreach ($case in @("getfnative", "getfnative_v2", "selectkernel")) {
    Invoke-CpuProfile "assess-sweep-$case-r32" "assess_ext" `
        $case $SweepFrames 32 $SweepScript
    Invoke-MemoryTrace "memory-sweep-$case-r32" `
        $case $SweepFrames 32 $SweepScript
}

foreach ($requests in @(1, 8, 32)) {
    Invoke-MemoryTrace "memory-bicubic_b3-r$requests" `
        "bicubic_b3" 4096 $requests $KernelScript
}

if (-not $SkipEtw) {
    Invoke-EtwTrace "etw-bicubic_b3-r32" "bicubic_b3" `
        $KernelFrames 32 $KernelScript
    Invoke-EtwTrace "etw-sweep-getfnative-r32" "getfnative" `
        $SweepFrames 32 $SweepScript
}

Complete-Manifest
Write-RunLog "All requested profile jobs finished. Manifest: $ManifestPath"
