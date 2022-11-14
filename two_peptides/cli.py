
import click


@click.group()
def cli_main():
    pass

from .run import run_cmd
cli_main.add_command(run_cmd)

from .status import status_cmd
cli_main.add_command(status_cmd)

if __name__ == "__main__":
    cli_main()

