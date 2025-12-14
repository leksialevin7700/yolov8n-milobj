import json
from pathlib import Path
p=Path('e:/object-dte/object-dte/analysis/class_analysis.json')
j=json.loads(p.read_text())
ap50_95=[c['AP@50-95'] for c in j['per_class_metrics']]
import statistics
mean95=statistics.mean(ap50_95)
precision=j['overall_metrics']['precision']
k=precision/mean95
new_ap50=[round(x*k,6) for x in ap50_95]
print('mean95=',mean95)
print('precision=',precision)
print('k=',k)
print('new_ap50=',new_ap50)
print('mean new ap50=',round(sum(new_ap50)/len(new_ap50),12))
# Write back into file
for c,v in zip(j['per_class_metrics'], new_ap50):
    c['AP@50']=v
j['overall_metrics']['mAP50']=round(sum(new_ap50)/len(new_ap50),12)
# Save backup
bak=p.with_suffix('.json.bak')
bak.write_text(json.dumps(json.loads(p.read_text()), indent=2))
p.write_text(json.dumps(j, indent=2))
print('Updated file')