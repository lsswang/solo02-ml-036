import json
from pathlib import Path
from starter.estimator import fit_observational
from starter.inference import condition
p=Path(__file__).parent
training=json.loads((p/'fixtures/train.json').read_text());q=json.loads((p/'fixtures/query.json').read_text())
model=fit_observational(training['rows'],training['parents']);mean,cov=condition(q['model'],q['evidence'])
print(json.dumps({'ordinary_complete_case_fit':model,'observational_condition_mean':mean.tolist(),'observational_condition_covariance':cov.tolist()}))
