import os
from dataclasses import dataclass

import cv2
import itk
import numpy as np

# from utils.constants import SHAPE
SHAPE = (512, 512)

COMPONENT_TYPE_SWITCHER = {
    itk.CommonEnums.IOComponent_UCHAR: "unsigned char",
    itk.CommonEnums.IOComponent_CHAR: "char",
    itk.CommonEnums.IOComponent_USHORT: "unsigned short",
    itk.CommonEnums.IOComponent_SHORT: "short",
    itk.CommonEnums.IOComponent_UINT: "unsigned int",
    itk.CommonEnums.IOComponent_INT: "int",
    itk.CommonEnums.IOComponent_ULONG: "unsigned long",
    itk.CommonEnums.IOComponent_LONG: "long",
    itk.CommonEnums.IOComponent_FLOAT: "float",
    itk.CommonEnums.IOComponent_DOUBLE: "double",
}


def switcher(table: dict, value):
    return table.get(value, "Not Found")


@dataclass
class DicomImage:
    """
    Simple dicom image class to open a dicom series and load images+meta.
    """

    image: itk.Image = None

    def __eq__(self, other):
        if isinstance(other, DicomImage):
            return super().__eq__(other)
        elif isinstance(other, dict):
            for key, values in other.items():
                print("todo")

    def get_image_io(self, filename):
        """
        Detect automatically the image io module to use and return it.
        Args:
            filename: Path to an image path.

        Returns: The corresponding image_io.
        See Also: https://simpleitk.readthedocs.io/en/master/IO.html

        """
        image_io = itk.ImageIOFactory.CreateImageIO(
            filename, itk.CommonEnums.IOFileMode_ReadMode
        )
        image_io.SetFileName(filename)
        image_io.ReadImageInformation()

        return image_io

    def read(self, filenames, pixel_type=None, number_of_dimension=None):
        """
        Read multiple dicom files from the same series.
        Args:
            filenames: Dicom path (list).
            pixel_type: itk pixel type.
            number_of_dimension: Number of axis (2-3).

        Returns: itk.Image series read from filenames with pixel_type.
        See Also: https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage/Documentation.html, https://itk.org/ITKExamples/src/IO/ImageBase/ReadUnknownImageType/Documentation.html#

        """
        if len(filenames) >= 1:

            image_io = self.get_image_io(filenames[0])
            metad = image_io.GetMetaDataDictionary()

            #  Get information from image
            # print(pixel_type)
            if not pixel_type:
                pixel_type = itk.ctype(
                    switcher(COMPONENT_TYPE_SWITCHER, image_io.GetComponentType())
                )

            if not number_of_dimension:
                number_of_dimensions = image_io.GetNumberOfDimensions()
            self.image = itk.Image[pixel_type, number_of_dimensions].New()
            reader = itk.ImageSeriesReader[self.image].New()
            reader.SetImageIO(image_io)
            reader.SetFileNames(filenames)
            reader.ForceOrthogonalDirectionOff()
            reader.Update()
            self.image.Graft(reader.GetOutput())
            return self.image

    def as_numpy_array(self, pixel_type=np.uint8, max_pixel_val=255, view=True):
        if view:
            image = itk.GetArrayViewFromImage(self.image)
        else:
            image = itk.GetArrayFromImage(self.image)
        normalized_image = np.zeros(SHAPE)
        normalized_image = cv2.normalize(
            image, normalized_image, 0, max_pixel_val, cv2.NORM_MINMAX
        )

        return normalized_image.astype(pixel_type)
        # return image


class DirectDicomImport:
    """
    Parse a directory recursively and import all dicom files as an itk image. Store DicomImage objects in self.images list.
    """

    def __init__(
        self,
        image_folder: str,
        verbosity: bool = False,
        seris_refiners: list = ["0008|0021"],
    ):
        """
        Args:
            image_folder: Path to a folder containing 1 or multiple series.
            verbosity: (boolean) Display warning when importing image (default False).
            seris_refiners: Dicom tag to define boundaries between different series.
        """
        self.image_folder = image_folder
        self.series_refiners = seris_refiners
        self._verbosity = verbosity
        self.namesGenerator = 0
        self.images = []
        self.images_id = []

    @property
    def namesGenerator(self):
        return self._namesGenerator

    @namesGenerator.setter
    def namesGenerator(self, value):
        """
        NamesGenerator is the module to subdivise all dicom files contained in a folder into series. it is also generating the filenames.
        Args:
            value: Not Used.

        Returns: NameGenerators with Series Restriction defined as refiners.

        """
        self._namesGenerator = itk.GDCMSeriesFileNames.New()
        self._namesGenerator.SetUseSeriesDetails(True)
        if self.series_refiners:
            for series_refiner in self.series_refiners:
                self._namesGenerator.AddSeriesRestriction(series_refiner)
        self._namesGenerator.SetGlobalWarningDisplay(self._verbosity)

    def recursive_directory_parser(self):
        """
        Parse all subdirectories and directories and yield root path if a file is found.
        Returns:

        """
        for root, dirs, files in os.walk(self.image_folder):
            if files:
                yield root

    def read_files(self):
        """
        Read dicom files if files is dicom and add DicomImage object into list of images.
        Returns:

        """
        for path in self.recursive_directory_parser():
            series_uids = self.get_series_uid(path)
            for series_uid in series_uids:
                filenames = self.get_filenames(series_uid)
                dcm_image = DicomImage()
                dcm_image.read(filenames)
                self.images.append(dcm_image)
                for i in filenames:
                    self.images_id.append(os.path.basename(i).split(".")[-1])

    def get_series_uid(self, folder):
        self.namesGenerator.SetDirectory(folder)
        return self.namesGenerator.GetSeriesUIDs()

    def get_filenames(self, series_uid):
        return self.namesGenerator.GetFileNames(series_uid)


if __name__ == "__main__":
    filename = "E:/Projects/Shoulder/data/images/a0001 aishoulder0001/arthro-irm-epaule-d/MR ax-pd-fs-propeller/"
    folder = (
        "E:/Projects/Shoulder/data/images/a0001 aishoulder0001/arthro-irm-epaule-d/"
    )
    # imageio = itk.ImageIOFactory.CreateImageIO(filename, itk.CommonEnums.IOFileMode_ReadMode)
    # imageio.SetFileName(filename)
    # imageio.ReadImageInformation()
    # img = itk.image_file_reader(FileName="E:/Projects/Shoulder/data/images/a0001 aishoulder0001/arthro-irm-epaule-d/MR ax-pd-fs-propeller/MR000001.dcm", ImageIO=itk.GDCMImageIO.New())
    # print(imageio.GetPixelType())
    # DDI = DirectDicomImport(folder)
    # DDI.read_files()
    # series = DDI.get_series_uid(folder)
    # fileNames =DDI.namesGenerator.GetFileNames(series[0])
    # DI = DicomImage()
    # DI.read(fileNames)
    dcm_image = DicomImage()
    print(dcm_image.read(["/users/ch_mariiavidmuk/shoulderai_tangent/testing_folder/1.2.840.113619.2.322.4120.14256054.13815.1460700213.376"]))
    # print(dcm_image.read(["/data/soin/shoulder_ai/data/2024/DATA/dicom/new_dicom_2024/F0004/T1_sag/B9F67662"]))