.DEFAULT_GOAL := help
ALGO ?= erl

.PHONY: help run clean run run-all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

run: ## Run algorithm (ALGO=erl by default, e.g. make run ALGO=td3)
	uv run python entry_point.py algorithm=$(ALGO) $(ARGS)


run-all: ## Run TD3, ERL and SC-ERL for seeds 0,1,2

	uv run python entry_point.py algorithm=td3 seed=0 wandb.name=td3 wandb.tags=[Pendulum,TD3,baseline]
	uv run python entry_point.py algorithm=erl seed=0 wandb.name=erl wandb.tags=[Pendulum,ERL,baseline]
	uv run python entry_point.py algorithm=sc_erl seed=0 wandb.name=sc_erl wandb.tags=[Pendulum,SC_ERL,baseline]

clean: ## Clean outputs and Hydra logs
	rm -rf outputs .hydra