#!/usr/bin/env python3
"""Persist a same-host release CPU A/B, including frozen source and output proofs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import tarfile
import time


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def freeze(source, target):
    try:
        root = Path(subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', '--show-toplevel'],
            text=True, stderr=subprocess.DEVNULL).strip())
    except subprocess.CalledProcessError:
        root = None
    if root == source:
        names = subprocess.check_output([
            'git', '-C', str(source), 'ls-files', '--cached', '--others',
            '--exclude-standard', '-z']).decode().split('\0')
        paths = [source / name for name in names if name]
    else:
        paths = [p for p in source.rglob('*') if p.is_file() and '.git' not in p.parts]
    target.mkdir()
    for path in paths:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def manifest(source):
    return {p.relative_to(source).as_posix(): digest(p)
            for p in sorted(source.rglob('*')) if p.is_file()}


def build(source, build_dir, driver, executable, compiler, sdk, jobs, log):
    with log.open('w') as stream:
        command = ['cmake', '-S', str(source), '-B', str(build_dir),
                   '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                   '-DBUILD_TESTING=ON', '-DDSMVC_ENABLE_NATIVE_CPU_SIMD=ON',
                   '-DDSMVC_ENABLE_CUDA=OFF', '-DDSMVC_ENABLE_VULKAN=OFF',
                   '-DDSMVC_ENABLE_METAL=OFF', '-DDSMVC_BUILD_BENCHMARKS=OFF']
        if os.environ.get('DSMVC_VAPOURSYNTH_SDK'):
            command += ['-DDSMVC_VAPOURSYNTH_SDK=' + os.environ['DSMVC_VAPOURSYNTH_SDK']]
        if sdk:
            command += ['-DDSMVC_VAPOURSYNTH_INCLUDE_DIR=' + sdk]
        if os.environ.get('NSS_C4_PYTHON'):
            command += ['-DDSMVC_VS_PYTHON=' + os.environ['NSS_C4_PYTHON']]
        run(command, stdout=stream, stderr=subprocess.STDOUT)
        run(['cmake', '--build', str(build_dir), '--parallel', str(jobs)],
            stdout=stream, stderr=subprocess.STDOUT)
        run(['ctest', '--test-dir', str(build_dir), '--output-on-failure', '-j1'],
            stdout=stream, stderr=subprocess.STDOUT)
        has_avx512 = 'cpu_avx512_available' in (source / 'include/dsmvc/engine.hpp').read_text()
        command = [*compiler, '-O3', '-std=c++23', '-ffp-contract=off', '-pthread',
                   '-DDSMVC_BENCH_HAS_AVX512_API=' + str(int(has_avx512)),
                   '-I' + str(source / 'include'), str(driver),
                   str(build_dir / 'libdsmvc_engine.a'), '-o', str(executable)]
        run(command, stdout=stream, stderr=subprocess.STDOUT)
    return {'driver_command': command, 'driver_sha256': digest(executable),
            'engine_sha256': digest(build_dir / 'libdsmvc_engine.a'),
            'plugin_sha256': digest(build_dir / 'dsmvc.so'), 'avx512_api': has_avx512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='v0.1.2')
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--candidate-source', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--stable-seconds', type=float, default=1.0)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--sdk-include', default=os.environ.get('DSMVC_VAPOURSYNTH_INCLUDE_DIR', ''))
    args = parser.parse_args()
    if platform.system() != 'Linux' or args.samples < 1 or args.iterations < 1:
        parser.error('Linux and positive sample/iteration counts are required')
    if not 0.05 <= args.stable_seconds <= 2.0:
        parser.error('stable-seconds must be between 0.05 and 2.0')
    candidate = args.candidate_source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: output / (name + '-source') for name in ['baseline', 'candidate']}
    baseline_revision = None
    if args.baseline_source:
        freeze(args.baseline_source.resolve(), sources['baseline'])
    else:
        baseline_revision = subprocess.check_output([
            'git', '-C', str(candidate), 'rev-parse', args.baseline], text=True).strip()
        archive = output / 'baseline.tar'
        run(['git', '-C', str(candidate), 'archive', '--format=tar',
             '-o', str(archive), baseline_revision])
        sources['baseline'].mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(sources['baseline'], filter='data')
    freeze(candidate, sources['candidate'])
    source_manifests = {name: manifest(path) for name, path in sources.items()}
    write_json(output / 'source-manifests.json', source_manifests)
    metadata = {'baseline_requested': args.baseline, 'baseline_revision': baseline_revision,
                'source_fingerprints': {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                                        for name, value in source_manifests.items()},
                'compiler': subprocess.check_output([*shlex.split(os.environ.get('CXX', 'c++')), '--version'], text=True),
                'lscpu': subprocess.check_output(['lscpu'], text=True),
                'kernel': platform.platform(), 'samples': args.samples,
                'legacy_iterations': args.iterations, 'stable_seconds': args.stable_seconds,
                'input_offset_mod64': 16, 'source_fills_inside_timer': 0}
    write_json(output / 'metadata.json', metadata)
    driver = output / 'cpu_release_probe.cpp'
    shutil.copyfile(candidate / 'benchmarks/cpu_release_probe.cpp', driver)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    identity = {}
    for name, source in sources.items():
        identity[name] = build(source, output / (name + '-build'), driver,
                               output / ('probe-' + name), compiler,
                               args.sdk_include, args.jobs, output / (name + '-build.log'))
    for name in sources:
        identity[name]['capabilities'] = json.loads(subprocess.check_output(
            [str(output / ('probe-' + name)), '--capabilities'], text=True))
        if not identity[name]['capabilities']['avx2']:
            raise RuntimeError('AVX2 is required on both sides')
    write_json(output / 'binary-identities.json', identity)
    os.sched_setaffinity(0, {0})
    observations, summaries = [], []
    proofs = output / 'proofs'
    proofs.mkdir()
    paths = [('avx2', 'avx2')]
    if identity['candidate']['capabilities']['avx512']:
        paths.append(('avx512' if identity['baseline']['capabilities']['avx512'] else 'avx2', 'avx512'))
    for axis in ['width', 'height']:
        for kernel in ['bilinear', 'bicubic', 'lanczos3', 'spline64']:
            for baseline_isa, candidate_isa in paths:
                case = f'{axis}-{kernel}-{baseline_isa}-to-{candidate_isa}'
                expected = None
                proof_cache = {}
                def probe(name, isa, iterations, label):
                    nonlocal expected
                    proof = proofs / f'{case}-{label}-{name}.bin'
                    command = [str(output / ('probe-' + name)), kernel, axis, isa,
                               '1', str(iterations), str(proof)]
                    start = time.monotonic()
                    result = subprocess.run(command, capture_output=True, text=True, timeout=29)
                    (output / f'{case}-{label}-{name}.log').write_text(result.stdout + '\n' + result.stderr)
                    if result.returncode:
                        raise RuntimeError((command, result.returncode, result.stderr))
                    row = json.loads(result.stdout)
                    assert row['samples'] == 1 and row['iterations'] == iterations
                    assert row['input_mod64'] == row['output_mod64'] == 16
                    assert row['timed_source_fills'] == 0
                    expected_bytes = (256 if axis == 'width' else 952) * 1692 * 4
                    if proof.stat().st_size != expected_bytes:
                        raise RuntimeError((case, 'incomplete output proof'))
                    signature = (digest(proof), row['input_hash'], row['actual_f64'])
                    if expected is None:
                        expected = signature
                    if signature != expected:
                        raise RuntimeError((case, name, 'baseline/candidate output or precision differs'))
                    retained = proof_cache.setdefault(name, proof)
                    if proof != retained:
                        proof.unlink()  # The identical buffer is retained once per binary/case.
                    ticks = row['cpu1_samples'][0]
                    idle = 100 * ticks['idle'] / ticks['total'] if ticks['total'] else None
                    record = {'case': case, 'variant': name, 'label': label, 'command': command,
                              'process_seconds': time.monotonic() - start, 'result': row,
                              'output_sha256': signature[0], 'proof_reference': str(retained.relative_to(output)),
                              'timed_cpu1_idle': idle}
                    observations.append(record)
                    write_json(output / 'samples.json', observations)
                    return record
                pilots = {name: probe(name, isa, args.iterations, 'pilot')
                          for name, isa in [('baseline', baseline_isa), ('candidate', candidate_isa)]}
                shortest = min(row['result']['median_ms'] / args.iterations for row in pilots.values())
                stable_iterations = max(args.iterations, math.ceil(args.stable_seconds * 1000 / shortest))
                for mode, iterations in [('legacy', args.iterations), ('stable', stable_iterations)]:
                    attempts = 3 if mode == 'stable' else 1
                    for attempt in range(attempts):
                        start = time.monotonic()
                        gains, idles = [], []
                        for pair in range(args.samples):
                            selected = {}
                            order = [('baseline', baseline_isa), ('candidate', candidate_isa)]
                            if pair % 2:
                                order.reverse()
                            for name, isa in order:
                                row = probe(name, isa, iterations, f'{mode}-a{attempt}-p{pair}')
                                selected[name] = row
                                idles.append(row['timed_cpu1_idle'])
                            gains.append(selected['baseline']['result']['median_ms'] /
                                         selected['candidate']['result']['median_ms'])
                        valid = mode == 'stable' and all(value is not None and value >= 99 for value in idles)
                        summary = {'case': case, 'mode': mode, 'attempt': attempt, 'iterations': iterations,
                                   'pairs': gains, 'speedup': statistics.median(gains),
                                   'timed_cpu1_idle_min': min((v for v in idles if v is not None), default=None),
                                   'environment_valid': valid, 'bit_exact': True,
                                   'actual_f64': pilots['baseline']['result']['actual_f64'],
                                   'wall_seconds': time.monotonic() - start}
                        summaries.append(summary)
                        write_json(output / 'summary.json', summaries)
                        print(json.dumps(summary), flush=True)
                        if summary['wall_seconds'] >= 30:
                            raise RuntimeError('one benchmark case exceeded the bounded 30-second budget')
                        if mode == 'legacy' or valid:
                            break
    for name, source in sources.items():
        if manifest(source) != source_manifests[name]:
            raise RuntimeError('frozen source changed during benchmark')
    (output / 'DONE').touch()


if __name__ == '__main__':
    main()
