import argparse
from typing import Optional


class ObjectWithArgparsedArgs:
    """Class of objects with arguments defined by `argparse`.

    If `args` is not given, all arguments are set using `argparse`.

    Args:
        args (argparse.Namespace, optional): Arguments.
    """

    def __init__(self, args: Optional[argparse.Namespace] = None, **kwargs):
        if args is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                allow_abbrev=False)
            self._default_args = self._define_args_defaults()
            self._add_args(parser)
            args = parser.parse_args()
        self._check_args(args)
        self._args = args

    @classmethod
    def _define_args_defaults(cls) -> dict:
        """Defines defaults of command-line arguments for this class.

        A subclass must override this method only if 1) the subclass introduces
        new arguments for which defaults should be set, or 2) the defaults
        defined here have to be updated.

        All subclasses that override this method *must* call the super method
        at the beginning of the overriding implementation. The returned
        dictionary can be updated before being returned.

        Returns:
            defaults (dict): a dictionary of argument names and default values.
        """
        return {}

    def _add_args(self, parser: argparse.ArgumentParser) -> None:
        """Adds class-specific arguments to the argument parser.

        Subclasses overriding this methods *must* call the super method.

        Args:
            parser (argparse.ArgumentParser): Argument parser automatically
                created when initializing this object.
        """
        pass

    @classmethod
    def _check_args(cls, args: argparse.Namespace) -> None:
        """Checks arguments relevant to this class.

        If a subclass overrides this method, it should check its own arguments,
        and it *must* call the super's implementation of this method.

        Args:
            args (argparse.Namespace)
        """
        pass

    @property
    def args(self) -> argparse.Namespace:
        """Arguments (configuration) of this object."""
        return self._args
