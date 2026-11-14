"""Pure identity-index algebra for Target1. No model, no outcome lookup."""
import numpy as np

def identity(e):
 return (e.get('segment_label',e.get('segment')),e['source_family'],int(e['source_episode']),e['source_role'],int(e['source_frame_index']))
def token_map(source_entries,target_entries,tokens_per_frame=16):
 """For each current token, position occupied by its identity in target layout."""
 s=[identity(e) for e in source_entries];t=[identity(e) for e in target_entries]
 if len(s)!=32 or len(set(s))!=32 or len(set(t))!=32 or set(s)!=set(t):raise ValueError('Need the same 32 unique semantic identities')
 rank={x:i for i,x in enumerate(t)}
 return (np.array([rank[x] for x in s],np.int32)[:,None]*tokens_per_frame+np.arange(tokens_per_frame,dtype=np.int32)).reshape(1,-1)
def reduction_order(source_entries,canonical_entries):
 return np.argsort(token_map(source_entries,canonical_entries)[0]).astype(np.int32)[None]
def age_key_positions(selected_steps,now,tokens_per_frame=16):
 """Deployable age map, no segment labels/future information; experimental."""
 s=np.asarray(selected_steps,dtype=np.int64)
 if s.shape!=(32,) or np.any(s>now) or np.any(s<0):raise ValueError('Need 32 observed timestamps <= now')
 # Fixed cap31 frames; duplicates in key positions are permitted, order is unchanged.
 frame_pos=31-np.minimum(now-s,31)
 return (frame_pos[:,None]*tokens_per_frame+np.arange(tokens_per_frame,dtype=np.int32)).reshape(1,512).astype(np.int32)
def bridge_metrics(baseline,clamped):
 """Three dependent history pairs aggregated WITHIN episode."""
 from itertools import combinations
 records=[]
 for s,t in combinations(['far','middle','recent'],2):
  db=(baseline[t][:16]-baseline[s][:16]).astype(float).ravel();dc=(clamped[t][:16]-clamped[s][:16]).astype(float).ravel()
  d0=float(np.abs(db).mean());d1=float(np.abs(dc).mean())
  records.append(dict(pair=s+'__'+t,native_mae=d0,clamped_mae=d1,absolute_reduction=d0-d1,relative_reduction=(d0-d1)/d0 if d0>0 else None,projection_removed=float((db-dc)@db/(db@db)) if db@db>0 else None,l2_residual=float(np.linalg.norm(dc))))
 return records
