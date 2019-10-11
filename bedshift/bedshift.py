""" Computing configuration representation """

import argparse
import logmuse
import logging
import os
from . import __version__

_LOGGER = logging.getLogger(__name__)

class _VersionInHelpParser(argparse.ArgumentParser):
    def format_help(self):
        """ Add version information to help text. """
        return "version: {}\n".format(__version__) + \
               super(_VersionInHelpParser, self).format_help()


def build_argparser():
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    banner = "%(prog)s - randomize BED files"
    additional_description = "\n..."

    parser = _VersionInHelpParser(
            description=banner,
            epilog=additional_description)

    parser.add_argument(
            "-V", "--version",
            action="version",
            version="%(prog)s {v}".format(v=__version__))

    parser.add_argument(
            "-b", "--bedfile", required=True,
            help="File path to bed file.")

    parser.add_argument(
            "-d", "--droprate", type=float, default=0.0,
            help="Droprate parameter")

    parser.add_argument(
            "-a", "--addrate", type=float, default=0.0,
            help="Addrate parameter")
    
    parser.add_argument(
            "-p", "--shiftrate", type=float, default=0.0,
            help="Shift prob.")

    parser.add_argument(
            "-m", "--mean", type=float, default=0.0,
            help="Mean shift")

    parser.add_argument(
            "-s", "--stdev", type=float, default=1.0,
            help="Stdev")




    return parser




def main():
    """ Primary workflow """

    parser = logmuse.add_logging_options(build_argparser())
    args, remaining_args = parser.parse_known_args()
    global _LOGGER
    _LOGGER = logmuse.logger_via_cli(args)

    _LOGGER.info("welcome to bedshift")
    _LOGGER.info("Shifting file: '{}'".format(args.bedfile))
    msg = """Params:
  droprate: {droprate}
  addrate: {addrate}
  shiftrate: {shiftrate}
  mean: {mean}
  stdev: {stdev}
  """

    _LOGGER.info(msg.format(
        droprate=args.droprate,
        addrate=args.addrate,
        shiftrate=args.shiftrate,
        mean=args.mean,
        stdev=args.stdev))

    if not args.bedfile:
        parser.print_help()
        _LOGGER.error("No bedfile given")
        sys.exit(1)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _LOGGER.error("Program canceled by user!")
        sys.exit(1)