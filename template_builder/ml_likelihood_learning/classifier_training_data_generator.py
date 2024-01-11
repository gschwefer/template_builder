"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
# pylint: disable=W0201
from email.policy import default
import sys

import gzip
import pickle

from tqdm.auto import tqdm
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
import numpy as np
import scipy

from argparse import ArgumentParser
from pathlib import Path

from astropy.table import QTable

from ctapipe.calib import CameraCalibrator, GainSelector
from ctapipe.core import QualityQuery, Tool, traits
from ctapipe.core.traits import List, classes_with_traits, Unicode, Bool, Int
from ctapipe.image import ImageCleaner, ImageModifier, ImageProcessor
from ctapipe.image.extractor import ImageExtractor
from ctapipe.reco.reconstructor import StereoQualityQuery

from ctapipe.fitting import lts_linear_regression

from ctapipe.io import (
    DataLevel,
    EventSource,
    SimTelEventSource,
    metadata,
)
from ctapipe.coordinates import (
    CameraFrame,
    NominalFrame,
    GroundFrame,
    TiltedGroundFrame,
)

from ctapipe.utils import EventTypeFilter
from ctapipe.image import dilate
from astropy.time import Time

from template_builder.nn_fitter import NNFitter
from template_builder.utilities import *
from template_builder.extend_templates import *

COMPATIBLE_DATALEVELS = [
    DataLevel.R1,
    DataLevel.DL0,
    DataLevel.DL1_IMAGES,
]

from .ml_utilities import shuffle_table

from art import tprint

__all__ = ["ProcessorTool"]


class TrainingDataGenerator(Tool):
    """
    Process data from lower-data levels up to DL1, including both image
    extraction and optinally image parameterization
    """

    name = "training-data-generator"

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
        help="Input sim_telarray simulation files",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*.simtel.zst",
        help="Give a specific file pattern for matching files in ``input_dir``",
    ).tag(config=True)

    compute_image = Bool(
        help="Compute image templates",
        default_value=False,
    ).tag(config=True)

    compute_time = Bool(
        help="Compute time templates",
        default_value=False,
    ).tag(config=True)

    shuffle = Bool(
        help="Already add a shuffle_column",
        default_value=False,
    ).tag(config=True)

    n_dilate = Int(
        help="Number of times to dilate the image",
        default_value=2,
    ).tag(config=True)

    max_events = Int(
        help="Number of events to process in the file",
        default_value=1000,
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    output_file = Unicode(default_value=".", help="base output file name").tag(
        config=True
    )

    aliases = {
        ("i", "input"): "TrainingDataGenerator.input_files",
        ("o", "output"): "TrainingDataGenerator.output_file",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "TrainingDataGenerator.max_events",
        "image-cleaner-type": "ImageProcessor.image_cleaner_type",
    }

    classes = (
        [
            CameraCalibrator,
            ImageProcessor,
            metadata.Instrument,
            metadata.Contact,
        ]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
        + classes_with_traits(QualityQuery)
        + classes_with_traits(ImageModifier)
        + classes_with_traits(EventTypeFilter)
    )

    def setup(self):
        if self.compute_image:
            self.image_computation = True
        else:
            self.image_computation = False
        if self.compute_time:
            self.time_computation = True
        else:
            self.time_computation = False

        if not self.compute_image and not self.compute_time:
            self.time_computation = True
            self.image_computation = True

        # setup components:
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

        self.focal_length_choice = "EFFECTIVE"
        try:
            self.event_source = EventSource(
                input_url=self.input_files[0],
                parent=self,
                focal_length_choice=self.focal_length_choice,
                max_events=self.max_events,
            )
        except RuntimeError:
            print("Effective Focal length not availible, defaulting to equivelent")
            self.focal_length_choice = "EQUIVALENT"
            self.event_source = EventSource(
                input_url=self.input_files[0],
                parent=self,
                focal_length_choice=self.focal_length_choice,
                max_events=self.max_events,
            )

        if not self.event_source.has_any_datalevel(COMPATIBLE_DATALEVELS):
            self.log.critical(
                "%s  needs the EventSource to provide either R1 or DL0 or DL1A data"
                ", %s provides only %s",
                self.name,
                self.event_source,
                self.event_source.datalevels,
            )
            sys.exit(1)

        self.calibrate = CameraCalibrator(
            parent=self, subarray=self.event_source.subarray
        )
        self.process_images = ImageProcessor(
            subarray=self.event_source.subarray, parent=self
        )
        self.event_type_filter = EventTypeFilter(parent=self)
        self.check_parameters = StereoQualityQuery(parent=self)

        # We need this dummy time for coord conversions later
        self.dummy_time = Time("2010-01-01T00:00:00", format="isot", scale="utc")
        if self.time_computation:
            self.timing_table = QTable(
                names=[
                    "obs_id",
                    "event_id",
                    "tel_id",
                    "true_energy",
                    "true_alt",
                    "true_az",
                    "true_core_x",
                    "true_core_y",
                    "true_x_max",
                    "x_max_diff",
                    "true_impact_distance",
                    "true_fov_offset",
                    "true_phi",
                    "pointing_alt",
                    "pointing_az",
                    "time_slope",
                ]
            )
        if self.image_computation:
            self.pixel_table = QTable(
                names=[
                    "obs_id",
                    "event_id",
                    "tel_id",
                    "pix_id",
                    "true_energy",
                    "true_alt",
                    "true_az",
                    "true_core_x",
                    "true_core_y",
                    "true_x_max",
                    "x_max_diff",
                    "true_impact_distance",
                    "true_fov_offset",
                    "true_phi",
                    "pix_x",
                    "pix_y",
                    "rot_pix_x",
                    "rot_pix_y",
                    "pointing_alt",
                    "pointing_az",
                    "charge",
                    "neighbor_charge",
                    "peak_time",
                    "peak_time_rel_to_array"
                ]
            )

    def start(self):
        """
        Process events
        """

        self.event_source.subarray.info(printer=self.log.info)

        for input_file in self.input_files:
            self.event_source = EventSource(
                input_url=input_file,
                parent=self,
                focal_length_choice=self.focal_length_choice,
                max_events=self.max_events,
            )
            self.point, self.tilt_tel = None, None

            for event in tqdm(
                self.event_source,
                desc=self.event_source.__class__.__name__,
                total=self.event_source.max_events,
                unit="events",
            ):
                self.calibrate(event)
                self.process_images(event)

                # When calculating alt we have to account for the case when it is rounded
                # above 90 deg
                alt_evt = event.simulation.shower.alt
                if alt_evt > 90 * u.deg:
                    alt_evt = 90 * u.deg

                # Get the pointing direction and telescope positions of this run
                if self.point is None:
                    alt = event.pointing.array_altitude
                    if alt > 90 * u.deg:
                        alt = 90 * u.deg

                    self.point = SkyCoord(
                        alt=alt,
                        az=event.pointing.array_azimuth,
                        frame=AltAz(obstime=self.dummy_time),
                    )

                    grd_tel = self.event_source.subarray.tel_coords
                    # Convert to tilted system
                    self.tilt_tel = grd_tel.transform_to(
                        TiltedGroundFrame(pointing_direction=self.point)
                    )

                # These values are keys for the template dict later
                pt_az = self.point.az.to(u.deg).value
                pt_alt = self.point.alt.to(u.deg).value

                # Create coordinate objects for source position
                src = SkyCoord(
                    alt=event.simulation.shower.alt.value * u.rad,
                    az=event.simulation.shower.az.value * u.rad,
                    frame=AltAz(obstime=self.dummy_time),
                )

                offset = self.point.separation(src).to_value(u.deg)

                energy = event.simulation.shower.energy

                # Calcualtion of xmax bin as a key for the template dicts later
                zen = 90 - event.simulation.shower.alt.to(u.deg).value
                # Store simulated Xmax
                mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

                # Calc difference from expected Xmax (for gammas).
                exp_xmax = xmax_expectation(energy.value)
                x_diff = mc_xmax - exp_xmax

                # Calculate core position in tilted system
                grd_core_true = SkyCoord(
                    x=np.asarray(event.simulation.shower.core_x) * u.m,
                    y=np.asarray(event.simulation.shower.core_y) * u.m,
                    z=np.asarray(0) * u.m,
                    frame=GroundFrame(),
                )

                self.tilt_core_true = grd_core_true.transform_to(
                    TiltedGroundFrame(pointing_direction=self.point)
                )

                # transform source direction into nominal system (where we store our templates)
                self.source_direction = src.transform_to(
                    NominalFrame(origin=self.point)
                )

                trigger_time=event.trigger.time

                for tel_id, dl1 in event.dl1.tel.items():
                    # First set the the last dict key missing, the impact distance
                    impact = (
                        np.sqrt(
                            np.power(
                                self.tilt_tel.x[tel_id - 1] - self.tilt_core_true.x, 2
                            )
                            + np.power(
                                self.tilt_tel.y[tel_id - 1] - self.tilt_core_true.y, 2
                            )
                        )
                        .to(u.m)
                        .value
                    )

                    geom = self.event_source.subarray.tel[tel_id].camera.geometry

                    fl = geom.frame.focal_length.to(u.m)

                    camera_coord = SkyCoord(
                        x=geom.pix_x,
                        y=geom.pix_y,
                        frame=CameraFrame(
                            focal_length=fl, telescope_pointing=self.point
                        ),
                    )

                    nom_coord = camera_coord.transform_to(
                        NominalFrame(origin=self.point)
                    )

                    x = nom_coord.fov_lon.to(u.deg)
                    y = nom_coord.fov_lat.to(u.deg)

                    phi = (
                        np.arctan2(
                            (self.tilt_tel.y[tel_id - 1] - self.tilt_core_true.y),
                            (self.tilt_tel.x[tel_id - 1] - self.tilt_core_true.x),
                        )
                        + 90 * u.deg
                    )

                    x_rot, y_rot = rotate_translate(
                        x,
                        y,
                        self.source_direction.fov_lon,
                        self.source_direction.fov_lat,
                        phi,
                    )
                    x_rot *= -1  # Reverse x axis to fit HESS convention
                    x_rot, y_rot = x_rot.ravel(), y_rot.ravel()

                    mask = dl1.image_mask

                    for _ in range(self.n_dilate):
                        mask = dilate(geom, mask)

                    # Apply mask
                    x_masked = x[mask].astype(np.float32)
                    y_masked = y[mask].astype(np.float32)

                    # Apply mask
                    x_rot_masked = x_rot[mask].astype(np.float32)
                    y_rot_masked = y_rot[mask].astype(np.float32)

                    pmt_signal = dl1.image
                    image = pmt_signal[mask].astype(np.float32)

                    pix_ids = np.arange(len(pmt_signal))
                    pix_ids_masked = pix_ids[mask]

                    nbor_indices = self.event_source.subarray.tel[
                        tel_id
                    ].camera.geometry.neighbor_matrix_sparse.indices
                    nbor_indptr = self.event_source.subarray.tel[
                        tel_id
                    ].camera.geometry.neighbor_matrix_sparse.indptr

                    peak_times = dl1.peak_time[mask]

                    tel_trigger_offset=event.trigger.tel[tel_id].time-trigger_time

                    if self.time_computation:
                        time_mask = np.logical_and(
                            peak_times > 0, np.isfinite(peak_times)
                        )
                        time_mask = np.logical_and(time_mask, image > 5)

                        if np.sum(time_mask) > 3:
                            time_slope = lts_linear_regression(
                                x=x_rot_masked[time_mask]
                                .to_value(u.deg)
                                .astype(np.float64),
                                y=peak_times[time_mask].astype(np.float64),
                                samples=3,
                            )[0][0]

                            new_time_row = [
                                event.index.obs_id,
                                event.index.event_id,
                                tel_id,
                                energy,
                                self.source_direction.fov_lon.to_value(u.deg),
                                self.source_direction.fov_lat.to_value(u.deg),
                                self.tilt_core_true.x.to_value(u.m),
                                self.tilt_core_true.y.to_value(u.m),
                                mc_xmax,
                                x_diff,
                                impact,
                                offset,
                                phi.to_value(u.deg),
                                pt_alt,
                                pt_az,
                                time_slope,
                            ]
                            # print(len(new_row))
                            self.timing_table.add_row(new_time_row)

                    if self.image_computation:
                        for j, (charge, time) in enumerate(zip(image, peak_times)):
                            neighbor_charge = np.mean(
                                pmt_signal[
                                    nbor_indices[
                                        nbor_indptr[pix_ids_masked[j]] : nbor_indptr[
                                            pix_ids_masked[j] + 1
                                        ]
                                    ]
                                ]
                            )
                            new_charge_row = [
                                event.index.obs_id,
                                event.index.event_id,
                                tel_id,
                                pix_ids_masked[j],
                                energy,
                                self.source_direction.fov_lon.to_value(u.deg),
                                self.source_direction.fov_lat.to_value(u.deg),
                                self.tilt_core_true.x.to_value(u.m),
                                self.tilt_core_true.y.to_value(u.m),
                                mc_xmax,
                                x_diff,
                                impact,
                                offset,
                                phi.to_value(u.deg),
                                x_masked[j],
                                y_masked[j],
                                x_rot_masked[j],
                                y_rot_masked[j],
                                pt_alt,
                                pt_az,
                                charge,
                                neighbor_charge,
                                time,
                                time+tel_trigger_offset.to_value("sec")*1e9
                            ]
                            self.pixel_table.add_row(new_charge_row)

    def finish(self):
        if self.time_computation:
            if self.shuffle:
                shuffled_timing_table = shuffle_table(self.timing_table, ["time_slope"])
                with open(
                    self.output_file
                    + "_dilate_{}.shuffle.time_gradient.pkl".format(self.n_dilate),
                    "wb",
                ) as of:
                    pickle.dump(shuffled_timing_table, of)
            else:
                with open(
                    self.output_file
                    + "_dilate_{}.time_gradient.pkl".format(self.n_dilate),
                    "wb",
                ) as of:
                    pickle.dump(self.timing_table, of)
        if self.image_computation:
            if self.shuffle:
                final_pixel_table = shuffle_table(
                    self.pixel_table, ["charge", "peak_time", "peak_time_rel_to_array"]
                )
                with open(
                    self.output_file
                    + "_dilate_{}.shuffle.pixel_charge.pkl".format(self.n_dilate),
                    "wb",
                ) as of:
                    pickle.dump(final_pixel_table, of)
            else:
                final_pixel_table = self.pixel_table
                with open(
                    self.output_file
                    + "_dilate_{}.pixel_charge.pkl".format(self.n_dilate),
                    "wb",
                ) as of:
                    pickle.dump(final_pixel_table, of)


def main():
    """run the tool"""
    print(
        "======================================================================================="
    )
    tprint("Time Training Data Generator")
    print(
        "======================================================================================="
    )

    tool = TrainingDataGenerator()
    tool.run()


if __name__ == "__main__":
    main()
