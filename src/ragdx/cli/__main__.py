"""Entry point for ``python -m ragdx.cli``.

The ``ragdx`` console script defined in ``pyproject.toml`` resolves
``ragdx.cli:main`` directly, but ``python -m ragdx.cli`` requires a
``__main__`` module on the package itself. Keep this one-liner so the
two invocation styles behave identically.
"""

from ragdx.cli import main

if __name__ == "__main__":
    main()
