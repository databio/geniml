import os
import sys
import yaml
import logging


class BedshiftYAMLHandler(object):
    def __init__(self, bedshifter, yaml_fp, logger=None):
        """
        Handles Bedshift perturbations from yaml files

        :param bedshift.Bedshift bedshifter: a Bedshift object
        :param str yaml_fp: path to yaml file
        :param logging.logger logger: logger object
        """
        self.bedshifter = bedshifter
        self.yaml_fp = yaml_fp
        if logger is not None:
            self._LOGGER = logger
        else:
            self._LOGGER = logging.getLogger("BedshiftYAMLHandler")

    def _print_sample_config(self):
        """
        bedshift_operations:
          - add:
            rate: 0.1
            mean: 100
            stdev: 20
          - drop_from_file:
            file: tests/test.bed
            rate: 0.1
            delimiter: \t
          - shift_from_file:
            file: bedshifted_test.bed
            rate: 0.3
            mean: 100
            stdev: 200
          - add_from_file:
            file: tests/small_test.bed
            rate: 0.2
          - cut:
            rate: 0.2
          - drop:
            rate: 0.30
          - shift:
            rate: 0.05
            mean: 100
            stdev: 200
          - merge:
            rate: 0.15
        """
        self._LOGGER.info(self._print_sample_config.__doc__)

    def _read_from_yaml(self, fp):
        with open(fp, "r") as yaml_file:
            config_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self._LOGGER.info("Loaded configuration settings from {}".format(fp))
        return config_data

    def handle_yaml(self):
        """
        Performs perturbations provided in the yaml config file in the order they were provided.
        """
        data = self._read_from_yaml(self.yaml_fp)
        operations = [operation for operation in data["bedshift_operations"]]
        num_changed = 0

        for operation in operations:
            ##### add #####
            if set(["add", "rate", "mean", "stdev"]) == set(list(operation.keys())):
                rate = operation["rate"]
                mean = operation["mean"]
                std = operation["stdev"]
                num_added = self.bedshifter.add(rate, mean, std)
                num_changed += num_added

            ##### add_from_file with no delimiter provided #####
            elif set(["add_from_file", "file", "rate"]) == set(list(operation.keys())):
                fp = operation["file"]
                if os.path.isfile(fp):
                    add_rate = operation["rate"]
                    num_added = self.bedshifter.add_from_file(fp, add_rate)
                    num_changed += num_added
                else:
                    self._logger.error("File '{}' does not exist.".format(fp))
                    sys.exit(1)

            ##### add_from_file with delimiter provided #####
            elif set(["add_from_file", "file", "rate", "delimiter"]) == set(
                list(operation.keys())
            ):
                fp = operation["file"]
                if os.path.isfile(fp):
                    add_rate = operation["rate"]
                    delimiter = operation["delimiter"]
                    num_added = self.bedshifter.add_from_file(fp, add_rate, delimiter)
                    num_changed += num_added
                else:
                    self._logger.error("File '{}' does not exist.".format(fp))
                    sys.exit(1)

            ##### drop #####
            elif set(["drop", "rate"]) == set(list(operation.keys())):
                rate = operation["rate"]
                num_dropped = self.bedshifter.drop(rate)
                num_changed += num_dropped

            ##### drop_from_file with no delimiter provided #####
            elif set(["drop_from_file", "file", "rate"]) == set(list(operation.keys())):
                fp = operation["file"]
                if os.path.isfile(fp):
                    drop_rate = operation["rate"]
                    num_dropped = self.bedshifter.drop_from_file(fp, drop_rate)
                    num_changed += num_dropped
                else:
                    self._LOGGER.error("File '{}' does not exist.".format(fp))
                    sys.exit(1)

            ##### drop_from_file with delimiter provided #####
            elif set(["drop_from_file", "file", "rate", "delimiter"]) == set(
                list(operation.keys())
            ):
                fp = operation["file"]
                if os.path.isfile(fp):
                    drop_rate = operation["rate"]
                    delimiter = operation["delimiter"]
                    num_dropped = self.bedshifter.drop_from_file(
                        fp, drop_rate, delimiter
                    )
                    num_changed += num_dropped
                else:
                    self._LOGGER.error("File '{}' does not exist.".format(fp))
                    sys.exit(1)

            ##### shift #####
            elif set(["shift", "rate", "mean", "stdev"]) == set(list(operation.keys())):
                rate = operation["rate"]
                mean = operation["mean"]
                std = operation["stdev"]
                num_shifted = self.bedshifter.shift(rate, mean, std)
                num_changed += num_shifted

            ##### shift_from_file #####
            elif set(["shift_from_file", "file", "rate", "mean", "stdev"]) == set(
                list(operation.keys())
            ):
                fp = operation["file"]
                if os.path.isfile(fp):
                    rate = operation["rate"]
                    mean = operation["mean"]
                    std = operation["stdev"]
                    num_shifted = self.bedshifter.shift_from_file(fp, rate, mean, std)
                    num_changed += num_shifted
                else:
                    self._LOGGER.error("File '{}' does not exist.".format(fp))
                    sys.exit(1)

            ##### shift_from_file with delimiter provided #####
            elif set(
                ["shift_from_file", "file", "rate", "mean", "stdev", "delimiter"]
            ) == set(list(operation.keys())):
                fp = operation["file"]
                if os.path.isfile(fp):
                    rate = operation["rate"]
                    mean = operation["mean"]
                    std = operation["stdev"]
                    delimiter = operation["delimiter"]
                    num_shifted = self.bedshifter.shift_from_file(
                        fp, rate, mean, std, delimiter
                    )
                    num_changed += num_shifted
                else:
                    self._LOGGER.error("File '{}' does not exist.".format(fp))
                    sys.exit(1)

            ##### cut #####
            elif set(["cut", "rate"]) == set(list(operation.keys())):
                rate = operation["rate"]
                num_cut = self.bedshifter.cut(rate)
                num_changed += num_cut

            ##### merge #####
            elif set(["merge", "rate"]) == set(list(operation.keys())):
                rate = operation["rate"]
                num_merged = self.bedshifter.merge(rate)
                num_changed += num_merged

            else:
                self._LOGGER.error(
                    "\n\nInvalid settings entered in the config file. Please refer to the example below.\n\n"
                )
                self._print_sample_config()
                sys.exit(1)

        return num_changed
