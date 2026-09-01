from dataclasses import dataclass, field
from os.path import exists


class FieldFormat:
    """
    Automatic field formatter.
    """

    def __init__(self, formatter, value):
        """
        Args:
            formatter: (callable) Func to format the input value.
            value: Input value.
        """
        self.formatter = formatter
        self.value = value

    def __setattr__(self, key, value):
        """
        Override the attribute set method to apply the format func before the assignement.
        Args:
            key: (str) Attribute name.
            value: (any) Value with/without format.

        Returns:

        """
        if key == "formatter":
            self.__dict__[key] = value
        else:
            if value == '"T=0"':
                value = 0
            elif value == '"TM=4"':
                value = 4
            elif value == "TM=1" or value == '"T=1"':
                value = 1
            self.__dict__[key] = self.formatter(value)


@dataclass
class OsirixAnnotation:
    """
    Osirix annotation as defined in the documentation. Imported as a CSV file.
    """

    image_no: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    roi_no: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    roi_mean: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_min: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_max: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_total: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_dev: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_name: FieldFormat = field(default_factory=lambda: FieldFormat(str, 0))
    roi_center_X: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_center_Y: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_center_Z: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    length: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    area: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_type: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    num_of_points: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    mm_X: list = None
    mm_Y: list = None
    mm_Z: list = None
    px_X: list = None
    px_Y: list = None

    def from_csv_line(self, annotation: list):
        """
        From one csv line, update the annotation with new values.
        Args:
            annotation: List of splitted csv line.
        """
        (
            self.image_no.value,
            self.roi_no.value,
            self.roi_mean.value,
            self.roi_min.value,
            self.roi_max.value,
            self.roi_total.value,
            self.roi_dev.value,
            self.roi_name.value,
            self.roi_center_X.value,
            self.roi_center_Y.value,
            self.roi_center_Z.value,
            self.length.value,
            self.area.value,
            self.roi_type.value,
            self.num_of_points.value,
        ) = annotation[0:15]

        if self.mm_X is None:
            self.mm_X = list()
            self.mm_Y = list()
            self.mm_Z = list()
            self.px_X = list()
            self.px_Y = list()

        for idx in range(15, len(annotation)):
            if annotation[idx] == '':
                annotation[idx] = 0
            if idx % 5 == 0:
                self.mm_X.append(float(annotation[idx]))
            elif idx % 5 == 1:
                self.mm_Y.append(float(annotation[idx]))
            elif idx % 5 == 2:
                self.mm_Z.append(float(annotation[idx]))
            elif idx % 5 == 3:
                self.px_X.append(float(annotation[idx]))
            elif idx % 5 == 4:
                self.px_Y.append(float(annotation[idx]))


class OsirixAnnotationList:
    """
    List of osirix Annotation.
    """

    def __init__(self):
        self.annotations: list[OsirixAnnotation] = []

    def check_file(self, file_path, extension):
        """Check if file exist.
        :param file_path: File path to the tested file.
        :param extension: Extension of the file.
        :return: True if file exist. False otherwise.
        """
        return exists(file_path) and file_path.endswith(extension)

    def load_from_csv(self, csv_filepath: str):
        """
        Load Osirix annotation from a csv file.
        Args:
            csv_filepath: Path to the annotation csv.
        """
        if not self.check_file(csv_filepath, "csv"):
            raise FileNotFoundError(csv_filepath)

        with open(csv_filepath, "r") as f:
            raw_data = f.read()
        raw_data = raw_data.split("\n")
        for index, annotation in enumerate(raw_data):

            if index > 0:  # don't read first line as it is column names
                annotation = annotation.split(",")

                if (
                    len(annotation) > 1
                ):  # caused by split('/n'), the last element can be empty

                    self.annotations.append(self.process_item(annotation))

    def process_item(self, annotation: list):
        """
        Process a line in the csv file. First 15 lines corresponds to metadata (like mean value, area, etc..).
        and all other lines corresponds to the points location in mm (mm_x, mm_y, mm_z) and in pixels (px_x, px_y).

        Args:
            annotation: One splitted csv line.

        Returns: (OsirixAnnotation) Object with loaded annotation.

        """
        current_annotation = OsirixAnnotation()
        current_annotation.from_csv_line(annotation)

        return current_annotation


@dataclass
class NewOsirixAnnotation:
    """
    New Osirix annotation class to handle the new CSV format.
    """
    image_no: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    roi_no: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    roi_mean: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_median: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_min: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_max: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_total: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_dev: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_name: FieldFormat = field(default_factory=lambda: FieldFormat(str, ""))
    roi_center_X: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_center_Y: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_center_Z: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    length_cm: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    length_pix: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    area_cm2: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    area_pix2: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    radius_width_cm: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    radius_height_cm: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    radius_width_pix: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    radius_height_pix: FieldFormat = field(default_factory=lambda: FieldFormat(float, 0))
    roi_type: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    sop_instance_uid: FieldFormat = field(default_factory=lambda: FieldFormat(str, ""))
    series_instance_uid: FieldFormat = field(default_factory=lambda: FieldFormat(str, ""))
    study_instance_uid: FieldFormat = field(default_factory=lambda: FieldFormat(str, ""))
    num_of_points: FieldFormat = field(default_factory=lambda: FieldFormat(int, 0))
    mm_X: list = None
    mm_Y: list = None
    mm_Z: list = None
    px_X: list = None
    px_Y: list = None

    def from_csv_line(self, annotation: list):
        """
        From one csv line, update the annotation with new values.
        Args:
            annotation: List of splitted csv line.
        """
        (
            self.image_no.value,
            self.roi_no.value,
            self.roi_mean.value,
            self.roi_median.value,
            self.roi_min.value,
            self.roi_max.value,
            self.roi_total.value,
            self.roi_dev.value,
            self.roi_name.value,
            self.roi_center_X.value,
            self.roi_center_Y.value,
            self.roi_center_Z.value,
            self.length_cm.value,
            self.length_pix.value,
            self.area_cm2.value,
            self.area_pix2.value,
            self.radius_width_cm.value,
            self.radius_height_cm.value,
            self.radius_width_pix.value,
            self.radius_height_pix.value,
            self.roi_type.value,
            self.sop_instance_uid.value,
            self.series_instance_uid.value,
            self.study_instance_uid.value,
            self.num_of_points.value,
        ) = annotation[0:25]

        if self.mm_X is None:
            self.mm_X = list()
            self.mm_Y = list()
            self.mm_Z = list()
            self.px_X = list()
            self.px_Y = list()

        for idx in range(25, len(annotation)):
            value = annotation[idx]
            if value:  # Check if the value is not an empty string
                if idx % 5 == 0:
                    self.mm_X.append(float(value))
                elif idx % 5 == 1:
                    self.mm_Y.append(float(value))
                elif idx % 5 == 2:
                    self.mm_Z.append(float(value))
                elif idx % 5 == 3:
                    self.px_X.append(float(value))
                elif idx % 5 == 4:
                    self.px_Y.append(float(value))

class NewOsirixAnnotationList:
    """
    List of new osirix annotations.
    """

    def __init__(self):
        self.annotations: list[NewOsirixAnnotation] = []

    def load_from_csv(self, csv_filepath: str):
        """
        Load new Osirix annotations from a string containing CSV data.
        Args:
            csv_data: String containing the annotation CSV.
        """
        with open(csv_filepath, "r") as f:
            raw_data = f.read()
        raw_data = raw_data.split("\n")
        for index, annotation in enumerate(raw_data):
            if index > 0:  # don't read first line as it is column names
                annotation = annotation.split(",")
                if len(annotation) > 1:  # avoid empty lines
                    self.annotations.append(self.process_item(annotation))

    def process_item(self, annotation: list):
        """
        Process a line in the csv data. First 23 lines correspond to metadata,
        and all other lines correspond to the points location in mm (mm_x, mm_y, mm_z) and in pixels (px_x, px_y).

        Args:
            annotation: One splitted csv line.

        Returns: (NewOsirixAnnotation) Object with loaded annotation.
        """
        current_annotation = NewOsirixAnnotation()
        current_annotation.from_csv_line(annotation)
        return current_annotation



if __name__ == "__main__":

    oa = OsirixAnnotationList()
    oa.load_from_csv(
        "E:/Projects/Shoulder/data/annotations/raw_data/all/AISHOULDER0001 - A0001/Cor T1 FS PROP/Cor T1 FS PROP.csv"
    )
    print(type(oa.annotations[0].roi_center_X.value))
