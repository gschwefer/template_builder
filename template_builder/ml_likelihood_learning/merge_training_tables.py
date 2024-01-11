from ctapipe.core import Tool, traits

from ctapipe.core.traits import List, Unicode, Bool


from argparse import ArgumentParser
from pathlib import Path

import sys
import gzip
import pickle
import warnings

from astropy import table

from .ml_utilities import shuffle_table


class TrainingDataMerger(Tool):

    """
    Tool to merge multiple astropy tables with ml training data.
    This is useful mostly because creating multiple subsets in parallel is much
    faster computationally than directly creating one large dictionary.
    """

    input_dir = traits.Path(
        default_value=None,
        help="Input directory",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input template dictionary files",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*run*",
        help="Give a specific file pattern for matching files in ``input_dir``",
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    output_file = Unicode(default_value=".", help="base output file name").tag(
        config=True
    )

    shuffle = Bool(
        help="Add a shuffle_column",
        default_value=False,
    ).tag(config=True)

    def setup(self):
        # Load in all input files
        args = self.parser.parse_args(self.extra_args)
        self.input_files.extend(args.input_files)
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))

        if not self.input_files:
            self.log.critical(
                "No input files provided, either provide --input-dir "
                "or input files as positional arguments"
            )
            sys.exit(1)

    def start(self):
        with open(self.input_files[0], "rb") as inp_file:
            this_table = pickle.load(inp_file)
        self.full_training_table = this_table
        # Open and unpickle the tempalte ditionaries
        for input_file in self.input_files[1:]:
            this_table = None
            with open(input_file, "rb") as inp_file:
                this_table = pickle.load(inp_file)
            self.full_training_table = table.vstack(
                [self.full_training_table, this_table], join_type="exact"
            )

    def finish(self):
        # Save the combined template
        if self.shuffle:
            if "charge" in self.full_training_table.keys():
                shuffled_training_table = shuffle_table(
                    self.full_training_table, ["charge","peak_time","peak_time_rel_to_array"]
                )
                outfile_handler = open(
                    self.output_file + ".shuffle.pixel_charge.pkl", "wb"
                )
                pickle.dump(shuffled_training_table, outfile_handler)
                outfile_handler.close()
            elif "time_slope" in self.full_training_table.keys():
                shuffled_training_table = shuffle_table(
                    self.full_training_table, ["time_slope"]
                )
                outfile_handler = open(
                    self.output_file + ".shuffle.time_gradient.pkl", "wb"
                )
                pickle.dump(shuffled_training_table, outfile_handler)
                outfile_handler.close()
        else:
            if "charge" in self.full_training_table.keys():
                outfile_handler = open(self.output_file + ".pixel_charge.pkl", "wb")
                pickle.dump(self.full_training_table, outfile_handler)
                outfile_handler.close()
            elif "time_slope" in self.full_training_table.keys():
                outfile_handler = open(self.output_file + ".time_gradient.pkl", "wb")
                pickle.dump(self.full_training_table, outfile_handler)
                outfile_handler.close()


def main():
    """run the tool"""

    tool = TrainingDataMerger()
    tool.run()


if __name__ == "__main__":
    main()
