
import click


@click.group()
def cli_main():
    pass

from .run import run_cmd

if __name__ == "__main__":
    cli_main()

