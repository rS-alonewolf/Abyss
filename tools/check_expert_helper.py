"""Hardware regression checks for an opt-in helper; no model or tokenizer download.

Requires Linux, an upstream baseline executable, a patched executable, an existing
native DeepSeek V4.1 pack, and a JSON file containing prompt token IDs (array or
{"ids": [...]}). The optional fault library is tools/expert_helper_faults.cpp.
This script only starts/stops its own engines; arrange exclusive GPUs beforehand.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


def run(args, name, ids):
    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('STRATA_GLM_EXPERT_HELPER', 'STRATA_GLM_EXPERT_HELPER_GIB', 'HELPER_FAULT_TEST', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpus,
               STRATA_GLM_NO_REUSE='1', STRATA_DS4_REUSE='0', STRATA_DS4_SLOT_KEEP='0',
               STRATA_GLM_USAGE='0',
               STRATA_GLM_NO_SPEC='1', STRATA_DS4_SPEC='0', STRATA_GLM_KDA_GRAPH='0',
               GLM_DUMP_H=str(folder / 'hidden.f32'))
    if name == 'zero':
        env['STRATA_GLM_EXPERT_HELPER'] = '0'
    elif name not in ('upstream', 'off'):
        env.update(STRATA_GLM_EXPERT_HELPER='1', STRATA_GLM_EXPERT_HELPER_GIB=str(args.helper_gib))
    if name == 'missing':
        env['CUDA_VISIBLE_DEVICES'] = args.gpus.split(',')[0]
    if name == 'oversize':
        env['STRATA_GLM_EXPERT_HELPER_GIB'] = '128'
    faults = ('allocation', 'prepare', 'input', 'completion', 'kernel')
    if name in faults:
        if args.fault_library is None:
            raise ValueError('fault modes require --fault-library')
        env.update(LD_PRELOAD=str(args.fault_library), HELPER_FAULT_TEST=name)
    exe = args.baseline if name == 'upstream' else args.exe
    command = [str(exe), '--serve', '--glm-pack', str(args.pack), '--max-context', str(args.context),
               '--prefill', str(args.prefill), '--layer-split', '0', '--spec', '0']
    lines = queue.Queue()
    results = []
    with (folder / 'stderr.log').open('w') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=log, text=True, bufsize=1, env=env)
        def reader():
            for line in proc.stdout:
                lines.put(line.rstrip())
            lines.put(None)
        threading.Thread(target=reader, daemon=True).start()
        def receive(deadline):
            line = lines.get(timeout=max(0.01, deadline-time.monotonic()))
            if line is None or line.startswith('ERR'):
                raise RuntimeError(f'{name}: {line or "unexpected EOF"}')
            return line
        try:
            deadline = time.monotonic()+args.timeout
            while not receive(deadline).startswith('READY '):
                pass
            for request in range(3):
                limit = max(512, args.tokens) if name == 'cancel' and request == 0 else args.tokens
                proc.stdin.write(f'GEN {limit} temperature=0 top_p=1 top_k=0 min_p=0 seed=42 spec=0 '
                                 + ','.join(map(str, ids))+'\n')
                proc.stdin.flush()
                tokens, stopped = [], False
                deadline = time.monotonic()+args.timeout
                while True:
                    line = receive(deadline)
                    if name == 'cancel-prefill' and request == 0 and line.startswith('PP ') and not stopped:
                        proc.stdin.write('STOP\n'); proc.stdin.flush(); stopped = True
                    elif line.startswith('T '):
                        tokens.append(int(line.split()[1]))
                        if name == 'cancel' and request == 0 and len(tokens) == 4:
                            proc.stdin.write('STOP\n'); proc.stdin.flush(); stopped = True
                    elif line.startswith('DONE '):
                        fields = line.split()
                        if name in ('cancel', 'cancel-prefill') and request == 0:
                            assert stopped and fields[5] == 'cancel', line
                            if name == 'cancel-prefill':
                                assert 'expert helper prepared' not in (folder/'stderr.log').read_text(), 'helper prepared after cancelled Prefill'
                        else:
                            assert tokens and fields[5] != 'cancel', line
                        assert int(fields[8]) == 0, ('unexpected reuse', line)
                        results.append({'tokens':tokens, 'done':line})
                        break
            proc.stdin.write('QUIT\n'); proc.stdin.flush(); proc.stdin.close()
            proc.wait(timeout=60)
            assert proc.returncode == 0, proc.returncode
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
    log = (folder/'stderr.log').read_text()
    if name in faults:
        assert 'FAULT_TEST injected '+name in log, 'fault was not exercised'
        assert re.search(r'CPU fallback|using the CPU lane', log), 'fallback was not reported'
    if name in ('missing', 'oversize'):
        assert 'using the CPU lane' in log, 'unavailable helper did not report fallback'
    if name in ('on', 'cancel', 'cancel-prefill'):
        assert any(int(n)>0 for n in re.findall(r'calls (\d+) experts',log)), 'helper did not execute'
    state = folder/'hidden.f32'
    assert state.exists() and state.stat().st_size > 0
    result = {'mode':name, 'results':results, 'hidden_sha256':hashlib.sha256(state.read_bytes()).hexdigest(),
              'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(), 'normal_exit':True}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    print(name, 'PASS', result['hidden_sha256'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('exe','baseline','pack','ids','output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--fault-library',type=Path)
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--helper-gib',type=float,default=13)
    parser.add_argument('--context',type=int,default=8192)
    parser.add_argument('--prefill',type=int,default=4096)
    parser.add_argument('--tokens',type=int,default=64)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--modes',default='upstream,off,zero,on,missing,oversize,allocation,prepare,input,completion,kernel,cancel,cancel-prefill')
    args = parser.parse_args()
    for key in ('exe','baseline','pack','ids','output','fault_library'):
        value = getattr(args,key)
        if value is not None:
            setattr(args,key,value.resolve())
    data = json.loads(args.ids.read_text())
    ids = data['ids'] if isinstance(data,dict) else data
    assert ids and all(type(n) is int and n >= 0 for n in ids) and len(ids) <= args.context
    args.output.mkdir(parents=True,exist_ok=True)
    results = [run(args,name,ids) for name in args.modes.split(',')]
    defaults = [r for r in results if r['mode'] in ('upstream','off','zero')]
    if len(defaults) > 1:
        assert len({r['hidden_sha256'] for r in defaults}) == 1, 'default hidden-state parity failed'
        assert all([x['tokens'] for x in r['results']] == [x['tokens'] for x in defaults[0]['results']] for r in defaults)
    (args.output/'passed.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
