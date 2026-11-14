#!/usr/bin/env python3
"""Actual RoboMME policy runner using supplied public intervention hooks.
CPU `plan` and `preflight` require no JAX. `run` requires the restored GPU stack.
The semantic anchor is privileged diagnostic information, NOT a deployment rule.
"""
from pathlib import Path
import argparse,json,os,sys,hashlib,subprocess,platform,traceback,importlib.util
import numpy as np
import importlib.metadata as md
from bridge_core import token_map,reduction_order,identity
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def package_snapshot():
 rows=[]
 for d in md.distributions():
  name=d.metadata.get('Name')
  if name:rows.append({'name':name,'version':d.version,'direct_url':d.read_text('direct_url.json')})
 return sorted(rows,key=lambda x:(x['name'].lower(),x['version'],x['direct_url'] or ''))
def tree(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
def check(args):
 missing=[m for m in ['jax','flax','sapien'] if importlib.util.find_spec(m) is None]
 issues=[]
 if missing:issues.append('Missing Python modules: '+', '.join(missing))
 if not Path('/dev/nvidiactl').exists():issues.append('No visible NVIDIA device /dev/nvidiactl')
 if not args.checkpoint or not (args.checkpoint/'params').exists():issues.append('Checkpoint params absent')
 if not args.repo or not (args.repo/'src/mme_vla_suite/models/integration/history_gemma.py').exists():issues.append('Patched model source absent')
 if not args.assignment or not args.assignment.is_file():issues.append('Frozen distractor assignment CSV absent')
 if args.repo:
  if not (args.repo/'third_party/robomme_benchmark/src/robomme').exists():issues.append('Pinned benchmark source absent; recording assets also required at runtime')
 return {'status':'BLOCKED' if issues else 'PREFLIGHT_PASSED_NOT_INFERENCE','issues':issues,'python':sys.version,'platform':platform.platform()}
def bootstrap(repo):
 os.chdir(repo)
 for p in ['src','scripts','examples/robomme','third_party/robomme_benchmark/src','packages/openpi-client/src']:sys.path.insert(0,str(repo/p))
def plan(args):
 records=json.load(open(args.constructions));out=[]
 for rec in records:
  entries={c:rec['conditions'][c]['retained'] for c in ['far','middle','recent']}
  out.append({'episode_key':rec['episode_key'],'conditions':{s:{'native_key_positions':list(range(512)),'canonical_reduction_order':reduction_order(entries[s],entries['far']).tolist(),'shared_key_positions':{t:token_map(entries[s],entries[t]).tolist() for t in entries}} for s in entries}})
 write(args.output,{'scope':'CPU mapping plan only; no model output','units':out});print(json.dumps({'status':'PLAN_WRITTEN','units':len(out)}))
def run(args):
 receipt=check(args)
 if receipt['issues']:raise RuntimeError(json.dumps(receipt))
 cfg=json.load(open(args.protocol));args.output.mkdir(parents=True,exist_ok=False)
 bootstrap(args.repo)
 import run_replication_a as ra
 from mme_vla_suite.policies import policy_config
 from mme_vla_suite.training import config
 import jax
 if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('JAX reports no GPU; no CPU fallback for formal inference')
 # Bind source/checkpoint/assignment before outcomes. No automatic downloads.
 ra.ASSIGNMENT_SOURCE=args.assignment
 if sha(args.assignment)!=ra.ASSIGNMENT_SOURCE_SHA256:raise RuntimeError('Assignment hash differs from original frozen assignment')
 checkpoint_files=tree(args.checkpoint);hist=args.checkpoint.parent/'history_config.txt'
 if hist.read_text().strip()!='perceptual-framesamp-modul.yaml':raise RuntimeError('Wrong checkpoint history config')
 if args.checkpoint_freeze:
  old=json.load(open(args.checkpoint_freeze))
  if checkpoint_files!=old['checkpoint_files']:raise RuntimeError('Checkpoint differs from archived Stage7 manifest')
 else:raise RuntimeError('--checkpoint-freeze required: restore original Stage7 freeze receipt')
 if sha(hist)!=old['checkpoint_history_config']['sha256']:raise RuntimeError('Checkpoint parent history_config bytes differ from Stage7 freeze')
 if any(not (args.repo/n).is_file() or sha(args.repo/n)!=h for n,h in old['source'].items()):raise RuntimeError('Model/source tree differs from archived Stage7 executed manifest')
 bench_commit=subprocess.check_output(['git','-C',str(args.repo/'third_party/robomme_benchmark'),'rev-parse','HEAD'],text=True).strip()
 if bench_commit!='856bc3a189d4172f3f47dbee4424d585f8d78db3':raise RuntimeError('Benchmark commit differs from frozen original')
 write(args.output/'freeze.json',{'protocol':cfg,'protocol_sha256':sha(args.protocol),'runner_sha256':sha(__file__),'core_sha256':sha(HERE/'bridge_core.py'),'source':tree(args.repo/'src'),'scripts':tree(args.repo/'scripts'),'checkpoint_files':checkpoint_files,'checkpoint_history_sha256':sha(hist),'benchmark_commit':bench_commit,'assignment_sha256':sha(args.assignment),'seed':args.seed,'phase':args.phase,'devices':[str(x) for x in jax.devices()],'environment':{k:os.environ.get(k) for k in ['XLA_FLAGS','JAX_DEFAULT_MATMUL_PRECISION','JAX_ENABLE_X64','JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','SAPIEN_RENDER_DEVICE','PYTHONPATH','NVIDIA_TF32_OVERRIDE']},'packages':package_snapshot()})
 if args.phase=='formal':
  if not args.engineering_receipt:raise RuntimeError('Formal run requires successful engineering receipt')
  eng=json.load(open(args.engineering_receipt))
  if eng.get('status')!='ENGINEERING_COMPLETE' or eng.get('protocol_sha256')!=sha(args.protocol) or eng.get('seed')!=args.seed:raise RuntimeError('Engineering receipt/protocol/seed mismatch')
  eng_freeze=json.load(open(args.engineering_receipt.parent/'freeze.json'));current_freeze=json.load(open(args.output/'freeze.json'))
  for k in ['source','scripts','checkpoint_files','checkpoint_history_sha256','benchmark_commit','assignment_sha256','seed','devices','environment','packages','runner_sha256','core_sha256','protocol_sha256']:
   if eng_freeze[k]!=current_freeze[k]:raise RuntimeError('Engineering/formal execution provenance differs: '+k)
 policy=policy_config.create_trained_policy(config.get_config('mme_vla_suite'),args.checkpoint,seed=args.seed);ra.validate_replication_a_policy(policy)
 if policy._model.integration_type!='modulation' or not policy.config.use_pos_emb or policy.config.use_state_emb:raise RuntimeError('Unexpected model settings')
 eps=cfg['engineering_episodes'] if args.phase=='engineering' else cfg['primary_episodes'];done=[]
 try:
  for family in cfg['families']:
   for episode in eps:
    key=f'{family}_ep{episode}';d=args.output/key;d.mkdir();env=d/'environment';env.mkdir()
    inputs=ra.build_replication_a_unit_inputs(family,episode,env,1300)
    plans={c:ra.build_row_intervention_plan(c,inputs['assemblies'],inputs['retained_manifest']) for c in ['far','middle','recent']};entries={c:plans[c]['retained_entries'] for c in plans}
    write(d/'input_manifest.json',{'task_goal':inputs['task_goal'],'assignment_row':inputs['assignment_row'],'retained_manifest':inputs['retained_manifest'],'plans':plans})
    arrays={};invariants=[]
    for c in plans:
     ass=inputs['assemblies'][c];pl=plans[c];policy.reset();policy.add_buffer(ra.pack_buffer(ass.images,ass.states,ass.exec_start_idx));before=ra.hash_buffer_feats(policy.mem_buffer._history_feats)
     over=d/(c+'_selection.json');write(over,{'step_idx':int(pl['step_idx']),'indices_to_load':pl['selected_indices']});os.environ['MME_FRAMESAMP_OVERRIDE_PATH']=str(over.resolve())
     order=reduction_order(entries[c],entries['far']);ident=np.arange(512,dtype=np.int32)[None];pf=np.arange(32,dtype=np.int32)
     # native and explicit-native identity branch check, no reduction permutation.
     def infer(name,**kw):
      trace=d/(name+'_trace.jsonl');os.environ['MME_FRAMESAMP_TRACE_PATH']=str(trace.resolve())
      result=policy.infer({**inputs['query_obs'],'reset_rng':True,**kw});a=np.asarray(result['actions']);assert a.shape==(20,8) and np.isfinite(a).all();arrays[name]=a;np.save(d/(name+'.npy'),a,allow_pickle=False)
      traces=[json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
      if len(traces)!=1 or traces[0]['indices_to_load']!=pl['selected_indices']:raise RuntimeError('Selected identity trace mismatch')
      return a
     native=infer(c+'__native',temporal_pos_override={'enabled':False});explicit=infer(c+'__explicit',temporal_pos_override={'enabled':False},mem_frame_permutation=pf,mem_k_positions=ident,mem_reduction_order=ident)
     if not exact(native,explicit):raise RuntimeError('Native versus explicit identity numerical path mismatch')
     arms=[(u,k) for u in plans for k in plans];rng=np.random.default_rng(cfg['order_seed']+episode+sum(map(ord,family+c)));rng.shuffle(arms)
     for u,k in arms:
      infer(c+'__u_'+u+'__k_'+k,temporal_pos_override={'enabled':True,'step_idx':int(pl['step_idx']),'position_map':pl['temporal_maps'][u]},mem_frame_permutation=pf,mem_k_positions=token_map(entries[c],entries[k]),mem_reduction_order=order)
     # Matched numerical baseline, native upstream/native positions.
     base=infer(c+'__canonical',temporal_pos_override={'enabled':False},mem_frame_permutation=pf,mem_k_positions=ident,mem_reduction_order=order)
     if not exact(base,arrays[c+'__u_'+c+'__k_'+c]):raise RuntimeError('Canonical identity temporal map mismatch')
     # Common randomized identity map and different per-condition shams.
     for label,seed in [('common_shuffle',4264402),('mismatched_shuffle',4264403+['far','middle','recent'].index(c))]:
      perm=np.random.default_rng(seed).permutation(32);anchor=[entries['far'][i] for i in perm]
      infer(c+'__'+label,temporal_pos_override={'enabled':False},mem_frame_permutation=pf,mem_k_positions=token_map(entries[c],anchor),mem_reduction_order=order)
     replay=infer(c+'__replay',temporal_pos_override={'enabled':False},mem_frame_permutation=pf,mem_k_positions=ident,mem_reduction_order=order)
     if not exact(base,replay) or before!=ra.hash_buffer_feats(policy.mem_buffer._history_feats):raise RuntimeError('Replay/cache invariant failure')
     write(d/(c+'_cache.json'),{'before':before,'after':ra.hash_buffer_feats(policy.mem_buffer._history_feats)})
     os.environ.pop('MME_FRAMESAMP_OVERRIDE_PATH',None);os.environ.pop('MME_FRAMESAMP_TRACE_PATH',None)
    # Both-clamped exact recovery is expected algebraically; record, never use as bridge endpoint.
    positive={c:exact(arrays['far__u_far__k_far'],arrays[c+'__u_far__k_far']) for c in plans}
    write(d/'engineering_controls.json',{'double_clamp_bitwise':positive,'scope':'expected tensor-equivalence control, not scientific bridge success'})
    if args.phase=='engineering' and not all(positive.values()):raise RuntimeError('Double-clamp engineering equality failed; inspect represented tensors before formal work')
    np.savez_compressed(d/'actions.npz',**arrays);write(d/'complete.json',{'status':'COMPLETE','files':{p.name:sha(p) for p in d.glob('*.npy')},'seed':args.seed});done.append(key)
  write(args.output/'complete.json',{'status':'ENGINEERING_COMPLETE' if args.phase=='engineering' else 'FORMAL_COMPLETE','units':done,'protocol_sha256':sha(args.protocol),'seed':args.seed})
 except BaseException:
  write(args.output/'failure.json',{'status':'FAILED_PRESERVED','traceback':traceback.format_exc(),'completed_units':done});raise
 finally:
  os.environ.pop('MME_FRAMESAMP_OVERRIDE_PATH',None);os.environ.pop('MME_FRAMESAMP_TRACE_PATH',None)
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','preflight','run']);p.add_argument('--repo',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--checkpoint-freeze',type=Path);p.add_argument('--assignment',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path,default=HERE.parent/'configs/target1_v1.json');p.add_argument('--constructions',type=Path,default=HERE.parent/'results/complete_episode_constructions.json');p.add_argument('--seed',type=int,default=42);p.add_argument('--phase',choices=['engineering','formal'],default='engineering');p.add_argument('--engineering-receipt',type=Path);a=p.parse_args()
 for name,value in vars(a).items():
  if isinstance(value,Path):setattr(a,name,value.resolve())
 if a.command=='plan':plan(a)
 elif a.command=='preflight':r=check(a);write(a.output,r);print(json.dumps(r,indent=2))
 else:run(a)
if __name__=='__main__':main()
