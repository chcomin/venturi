# optimize.py
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from train import ExperimentRunner, get_parser


def objective(trial):
    # 1. Get base args
    parser, _ = get_parser()
    args = parser.parse_args([]) # Parse defaults
    
    # 2. Sample Hyperparameters via Optuna
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.model_params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Update run name to avoid conflicts
    args.run_name = f"trial_{trial.number}"
    args.save_val_imgs = False # Disable heavy IO during optimization
    
    # 3. Instantiate Runner
    runner = ExperimentRunner(args)
    
    # 4. Inject Pruning Callback
    # This checks intermediate results and kills bad trials early
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    
    # 5. Run (Non-interactive mode)
    # This runs the full training loop with the sampled params
    metric = runner.run(interactive=False, extra_callbacks=[pruning_callback])
    
    return metric

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)