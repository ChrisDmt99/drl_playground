```bash
conda create -n drl_env python=3.10 -y

conda activate drl_env

```

Run policy agent:
```bash
set PYTHONPATH=C:\Users\chris\Desktop\Projects\DeepReinforcementLearning\drl-playground

python scripts/run_policy_agent.py --config configs/policy_agent.yaml
```

Run value-function prediction

```bash
set PYTHONPATH=C:\Users\chris\Desktop\Projects\DeepReinforcementLearning\drl-playground

python scripts/run_value_agent.py --config configs/value_agent.yaml
```