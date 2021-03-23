import argparse
from bedshift._version import __version__


class _VersionInHelpParser(argparse.ArgumentParser):
    def format_help(self):
        """ Add version information to help text. """
        return (
            "version: {}\n".format(__version__)
            + super(_VersionInHelpParser, self).format_help()
        )


def build_argparser():
    """
    Builds argument parser.

    :return: argparse.ArgumentParser
    """

    banner = "%(prog)s - randomize BED files"
    additional_description = "\n..."

    parser = _VersionInHelpParser(description=banner, epilog=additional_description)

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s {v}".format(v=__version__),
    )

    parser.add_argument("-b", "--bedfile", required=True, help="File path to bed file.")

    parser.add_argument(
        "-l",
        "--chrom-lengths",
        type=str,
        required=False,
        help="TSV text file with one row per chromosomes indicating chromosome sizes",
    )

    parser.add_argument(
        "-g",
        "--genome",
        type=str,
        required=False,
        help="Refgenie genome identifier (used for chrom sizes).",
    )

    parser.add_argument(
        "-d", "--droprate", type=float, default=0.0, help="Droprate parameter"
    )

    parser.add_argument(
        "-a", "--addrate", type=float, default=0.0, help="Addrate parameter"
    )

    parser.add_argument(
        "--addmean", type=float, default=320.0, help="Mean add region length"
    )

    parser.add_argument("--addstdev", type=float, default=30.0, help="Stdev add length")

    parser.add_argument("--addfile", type=str, help="Add regions from a bedfile")

    parser.add_argument(
        "--valid-regions",
        type=str,
        dest="valid_regions",
        help="valid regions in which regions can be randomly added",
    )

    parser.add_argument(
        "-s", "--shiftrate", type=float, default=0.0, help="Shift probability"
    )

    parser.add_argument("--shiftmean", type=float, default=0.0, help="Mean shift")

    parser.add_argument("--shiftstdev", type=float, default=150.0, help="Stdev shift")

    parser.add_argument("--shiftfile", type=str, help="Shift regions from a bedfile")

    parser.add_argument(
        "-c", "--cutrate", type=float, default=0.0, help="Cut probability"
    )

    parser.add_argument(
        "-m",
        "--mergerate",
        type=float,
        default=0.0,
        help="Merge probability. WARNING: will likely create regions that are thousands of base pairs long",
    )

    parser.add_argument("--dropfile", type=str, help="Drop regions from a bedfile")

    parser.add_argument(
        "-o",
        "--outputfile",
        type=str,
        help="output file name (including extension). if not specified, will default to bedshifted_{originalname}.bed",
    )

    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=1,
        help="the number of times to repeat the operation",
    )

    parser.add_argument(
        "-y",
        "--yaml-config",
        dest="yaml_config",
        type=str,
        help="run yaml configuration file",
    )

    return parser


param_msg = """Params:
  chrom.sizes file: {chromsizes}
  shift:
    shift rate: {shiftrate}
    shift mean distance: {shiftmean}
    shift stdev: {shiftstdev}
    shift regions from file: {shiftfile}
  add:
    rate: {addrate}
    add mean length: {addmean}
    add stdev: {addstdev}
    add regions from file: {addfile}
    valid regions: {valid_regions}
  cut rate: {cutrate}
  drop rate: {droprate}
    drop regions from file: {dropfile}
  merge rate: {mergerate}
  outputfile: {outputfile}
  repeat: {repeat}
  yaml_config: {yaml_config}
"""
