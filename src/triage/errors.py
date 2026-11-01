"""Errors the user can fix (bad option, missing key, frozen file). The CLI prints them cleanly."""


class UserError(Exception):
    """Something the person running the command can fix. The message says how."""
