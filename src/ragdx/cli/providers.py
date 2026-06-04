"""``ragdx providers`` -- list known LLM providers / scaffold YAML.

ragdx routes generation through LiteLLM and the ragas judge through
langchain-openai, both of which are OpenAI-protocol-compatible. So
swapping providers is a YAML edit, not a code change. This command
makes that obvious:

* ``ragdx providers list`` -- print the catalog as a table.
* ``ragdx providers template <name>`` -- emit a paste-ready
  ``RAGConfig`` YAML stub for the named provider.

See :mod:`ragdx.providers` for the catalog itself.
"""

from __future__ import annotations

import typer
from rich import print
from rich.table import Table

from ragdx.cli._app import app
from ragdx.providers import CATALOG

_providers_app = typer.Typer(
    help="List and scaffold configs for known LLM providers.",
    no_args_is_help=True,
)


@_providers_app.command("list")
def list_providers() -> None:
    """Print the known-providers catalog."""
    table = Table(title="ragdx provider catalog", show_lines=False)
    table.add_column("name", style="cyan")
    table.add_column("label")
    table.add_column("sample model")
    table.add_column("api_base")
    table.add_column("env var(s)")
    for name, spec in CATALOG.items():
        table.add_row(
            name,
            spec.label,
            spec.sample_model,
            spec.api_base or "(SDK default)",
            ", ".join(spec.env_vars) or "(none)",
        )
    print(table)
    print(
        "\n[dim]Get a paste-ready YAML stub via: "
        "[bold]ragdx providers template <name>[/bold][/dim]"
    )


@_providers_app.command("template")
def template(
    name: str = typer.Argument(
        ..., help="Provider name from the catalog (see `ragdx providers list`)."
    ),
    project_label: str = typer.Option(
        "my-rag", "--label",
        help="Name written into the YAML's ``name`` field.",
    ),
    output: str = typer.Option(
        "", "--output", "-o",
        help="Write to this file instead of stdout.",
    ),
) -> None:
    """Emit a paste-ready RAGConfig YAML stub for ``name``."""
    from pathlib import Path

    from ragdx.providers import provider_template

    if name not in CATALOG:
        raise typer.BadParameter(
            f"Unknown provider {name!r}. "
            f"Known: {', '.join(sorted(CATALOG))}."
        )
    yaml_text = provider_template(name, project_label=project_label)
    if output:
        Path(output).write_text(yaml_text, encoding="utf-8")
        print(f"[green]Wrote[/green] {output}")
    else:
        print(yaml_text)


# Register as a subcommand group on the main app.
app.add_typer(_providers_app, name="providers")


__all__ = ["list_providers", "template"]
