import wandb

api = wandb.Api()
runs = api.runs("evo_rl/ue_evo_rl_3")

wimle_runs = [r for r in runs if "wimle" in r.name.lower()]
print(f"Znalezionych wimle runów po nazwie: {len(wimle_runs)}")
for r in wimle_runs[:3]:
    print(f"\nNazwa: {r.name}")
    print(f"Config: {dict(r.config)}")
