from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.loggers import Logger
import yaml


def yaml_to_dict(yaml_string):
    # Log config file to Weights and Biases
    try:
        # Split the string into lines
        lines = yaml_string.strip().split('\n')

        # Check if the first line is a comment
        first_line_comment = None
        if lines[0].strip().startswith('#'):
            first_line_comment = lines[0].strip()
            # Remove the first line for YAML parsing
            yaml_string = '\n'.join(lines[1:])

        # Parse the YAML string
        yaml_dict = yaml.safe_load(yaml_string)

        # Add the comment as a key-value pair if it exists
        if first_line_comment:
            pos = first_line_comment.find("==")
            yaml_dict['lightning_version'] = first_line_comment[pos+2:]

        return yaml_dict
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams(yaml_to_dict(config))


def cli_main():
    cli = LightningCLI(
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
