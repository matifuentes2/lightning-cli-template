# Lightning CLI Template

This repository provides a simple initial setup to run Pytorch experiments in a clean and reproducible manner using Lightning CLI.

- Run `pip install -e .` in the root of the repository
- To use Weights and Biases Logger, run `wand login` in terminal. Then specify your project and run names in the `config/*.yaml` file.
- To train models, run `python src/train.py --config configs/example.yaml`
- Define your own `LightningModule`, `DataModule`, and Pytorch models as needed.

YAML files provide a ton of customization options to adjust training and store the hyperparameters used for each run.
