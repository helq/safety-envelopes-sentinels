import code

__all__ = ['interact']


def interact(locals: dict) -> None:  # type: ignore
    code.InteractiveConsole(locals=locals).interact()
