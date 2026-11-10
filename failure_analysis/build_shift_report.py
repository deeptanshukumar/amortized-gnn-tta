"""Expand the small report into per-shift tables/plots. No GPU or Modal required."""
from pathlib import Path
import argparse,json,html,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'gcn':'GCN','graphsage':'GraphSAGE','gat':'GAT'}
ARCHES=list(NAMES);METHODS=['IM-LoRA','A','B','C','D']

def label(row):
    family=row['family'];s=float(row['severity'])
    if family=='control':return 'Unchanged graph'
    if family=='novel_attribute':return f'Gaussian feature noise: {s:g} × channel SD'
    if family=='novel_structural':return f'Degree-biased edge removal: {s:.0%}'
    return {'attribute':'Feature-entry masking','structural':'Uniform edge removal','combined':'Feature masking + edge removal'}[family]+f': {s:.0%}'

def build(source,destination):
    source=Path(source);out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    d=pd.read_csv(source/'per_case_metrics.csv')
    keys=['split','architecture','seed','method','model_id','environment_id']
    assert not d.duplicated(keys).any()
    counts=d.groupby(['split','architecture','seed','method','environment_id']).size()
    assert counts.eq(6).all(),'Expected six GNN instances per seed/method/shift'
    seed_keys=['split','environment_id','family','severity','group','architecture','method','seed']
    seed=d.groupby(seed_keys,as_index=False).agg(accuracy=('accuracy','mean'),gain=('gain','mean'),frozen=('baseline_accuracy','mean'))
    stats=seed.groupby(seed_keys[:-1],as_index=False).agg(accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        gain_mean=('gain','mean'),gain_sd=('gain','std'),frozen_mean=('frozen','mean'),seed_count=('seed','nunique'))
    assert stats.seed_count.eq(3).all()
    seed.to_csv(out/'per_shift_per_seed.csv',index=False);stats.to_csv(out/'per_shift_numeric.csv',index=False)
    d.to_csv(out/'per_model_per_seed_cases.csv',index=False)
    # Pool repeated draws WITHIN each seed, not as extra independent seeds.
    severity_keys=['split','family','severity','group','architecture','method','seed']
    pooled=seed.groupby(severity_keys,as_index=False).agg(gain=('gain','mean'),accuracy=('accuracy','mean'),frozen=('frozen','mean'))
    pooled_stats=pooled.groupby(severity_keys[:-1],as_index=False).agg(gain_mean=('gain','mean'),gain_sd=('gain','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'))
    pooled.to_csv(out/'severity_per_seed.csv',index=False);pooled_stats.to_csv(out/'severity_summary.csv',index=False)
    table_rows=[];absolute_rows=[]
    environments=stats[['split','environment_id','family','severity','group']].drop_duplicates().sort_values(['split','group','family','severity','environment_id'])
    for e in environments.to_dict('records'):
        for arch in ARCHES:
            part=stats[(stats.split==e['split'])&(stats.environment_id==e['environment_id'])&(stats.architecture==arch)].set_index('method')
            row=dict(split=e['split'],environment_id=e['environment_id'],condition=label(e),group=e['group'],backbone=NAMES[arch],
                **{'Frozen accuracy (%)':f"{100*part.loc['Frozen','accuracy_mean']:.3f}"})
            absolute=dict(row)
            for m in METHODS:
                row[m+' gain (pp)']=f"{100*part.loc[m,'gain_mean']:+.3f} ± {100*part.loc[m,'gain_sd']:.3f}"
                absolute[m+' accuracy (%)']=f"{100*part.loc[m,'accuracy_mean']:.3f} ± {100*part.loc[m,'accuracy_sd']:.3f}"
            table_rows.append(row);absolute_rows.append(absolute)
    tables=pd.DataFrame(table_rows);pd.DataFrame(absolute_rows).to_csv(out/'all_shift_absolute_accuracies.csv',index=False)
    tables.to_csv(out/'all_shift_gain_tables.csv',index=False)
    notes='''Mean ± sample SD is across generator/IM seeds 202801, 202802, 202803, after averaging six fixed GNN instances within each seed. Random graph draws are listed separately; the severity summary averages draws within seed before calculating SD. Shared graphs and GNN instances are not independent datasets. IM-LoRA has 30 steps, lr .01, rank 4, and three initialization seeds. GAT uses best trained checkpoints instead of identity under the requested forced-adaptation policy. No new checkpoint selection, adaptation, training or test scoring was performed for this breakdown.'''
    pages=['<!doctype html><html><head><meta charset="utf-8"><title>Arxiv KL: every shift</title><style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ccd4dc;padding:8px;text-align:right}th{background:#edf2f7}td:first-child,th:first-child{text-align:left}img{max-width:100%}nav a{margin-right:20px}section{margin-top:36px}small{color:#536173}</style></head><body><h1>Controlled Arxiv KL results: every shift</h1><p>'+html.escape(notes)+'</p><nav><a href="#validation">Validation</a><a href="#test">Test</a><a href="#plots">Plots</a><a href="all_shift_gain_tables.csv">All tables CSV</a><a href="all_shift_absolute_accuracies.csv">Absolute accuracies CSV</a></nav>']
    markdown=['# Per-shift KL report','',notes,'']
    plots=[]
    for split in ['validation','test']:
        frame=tables[tables.split.eq(split)];frame.to_csv(out/f'{split}_all_shifts.csv',index=False)
        pages.append(f'<h2 id="{split}">{split.title()}</h2>')
        markdown.extend([f'## {split.title()}',''])
        envs=environments[environments.split.eq(split)]
        for e in envs.to_dict('records'):
            title=label(e);eid=e['environment_id'];part=frame[frame.environment_id.eq(eid)]
            cols=['backbone','Frozen accuracy (%)']+[m+' gain (pp)' for m in METHODS]
            pages.append('<section><h3>'+html.escape(title)+'</h3><small>'+html.escape(eid+' · '+e['group'])+'</small>'+part[cols].to_html(index=False,escape=True)+'</section>')
            markdown.extend(['### '+title,'',f'Environment: `{eid}`; group: {e["group"]}','',
                '| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |'])
            markdown += ['| '+' | '.join(map(str,r))+' |' for r in part[cols].to_numpy()];markdown.append('')
        # Each environment is shown, including each repeated random draw.
        fig,axes=plt.subplots(1,3,figsize=(17,max(4,.35*len(envs))))
        labels=[label(e)+' ['+e['environment_id'].replace('validation-','').replace('test-','')+']' for e in envs.to_dict('records')]
        for ax,arch in zip(axes,ARCHES):
            p=stats[(stats.split==split)&(stats.architecture==arch)&stats.method.isin(list('ABCD'))]
            matrix=p.pivot(index='environment_id',columns='method',values='gain_mean').reindex(envs.environment_id)[list('ABCD')]*100
            limit=max(float(np.abs(matrix.to_numpy()).max()),.01)
            im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(4),list('ABCD'));ax.set_yticks(range(len(matrix)),labels if arch=='gcn' else ['']*len(matrix));ax.set_title(NAMES[arch])
            for i in range(matrix.shape[0]):
                for j in range(4):
                    v=matrix.iloc[i,j];ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.65*limit else 'black')
            fig.colorbar(im,ax=ax,label='Mean gain (pp)',fraction=.04)
        fig.suptitle(split.title()+' — every shift/draw; generator gains');fig.tight_layout()
        name=f'{split}_every_shift_heatmap.png';fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig);plots.append(name)
        # Keep IM separate so negative baseline does not conceal tiny generator gains.
        fig,axes=plt.subplots(1,3,figsize=(17,max(4,.35*len(envs))))
        for ax,arch in zip(axes,ARCHES):
            p=stats[(stats.split==split)&(stats.architecture==arch)&stats.method.eq('IM-LoRA')].set_index('environment_id').reindex(envs.environment_id)
            ax.barh(np.arange(len(envs)),100*p.gain_mean,xerr=100*p.gain_sd,capsize=2,color='slateblue')
            ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(envs)),labels if arch=='gcn' else ['']*len(envs));ax.invert_yaxis();ax.set_title(NAMES[arch]);ax.set_xlabel('IM-LoRA gain (pp)')
        fig.suptitle(split.title()+' — IM-LoRA for every shift, mean ± seed SD');fig.tight_layout()
        name=f'{split}_im_every_shift.png';fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig);plots.append(name)
        fig,axes=plt.subplots(3,3,figsize=(15,11),squeeze=False)
        for i,arch in enumerate(ARCHES):
            for j,family in enumerate(['attribute','structural','combined']):
                ax=axes[i,j]
                for c in 'ABCD':
                    p=pooled_stats[(pooled_stats.split==split)&(pooled_stats.architecture==arch)&(pooled_stats.family==family)&(pooled_stats.method==c)].sort_values('severity')
                    ax.errorbar(100*p.severity,100*p.gain_mean,yerr=100*p.gain_sd,marker='o',label=c,capsize=3)
                ax.axhline(0,color='black',ls='--',lw=1);ax.set_title(NAMES[arch]+' / '+family);ax.set_xlabel('Corruption rate (%)');ax.set_ylabel('Gain (pp)');ax.grid(alpha=.2)
        axes[0,0].legend();fig.suptitle(split.title()+' — severity response (draws averaged within seed)');fig.tight_layout()
        name=f'{split}_severity_response.png';fig.savefig(out/name,dpi=160);plt.close(fig);plots.append(name)
    pages.append('<h2 id="plots">Plots</h2>')
    for name in plots:pages.append('<h3>'+html.escape(name.replace('_',' '))+'</h3><img src="'+name+'">')
    pages.append('</body></html>');(out/'index.html').write_text('\n'.join(pages));(out/'every_shift_tables.md').write_text('\n'.join(markdown))
    (out/'README.md').write_text('# Per-shift report\n\nOpen index.html for all 34 environment tables and six plots.\n\n'+notes+'\n\nGaussian noise strength is in channel-SD units, not feature-mask percentage. Degree-biased deletion is a separate operation from uniform removal. Absolute accuracy and raw per-model/per-seed values are included.\n')
    with zipfile.ZipFile(out/'shift_details.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name!='shift_details.zip':z.write(p,p.name)
    print('Built',len(environments),'environments,',len(tables),'backbone/shift rows,',len(plots),'plots; ZIP bytes',(out/'shift_details.zip').stat().st_size)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=Path(__file__).parent/'reports/arxiv-kl-report-001');parser.add_argument('--output',type=Path)
    a=parser.parse_args();build(a.source,a.output or a.source/'shift_details')
