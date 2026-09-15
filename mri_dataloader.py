"""Shared MRI loading for ADD classification and FairTMJ training.

Annotation rows use semicolon-separated fields. Zero-based columns 0, 1 and
3 contain patient ID, joint side and the ADD class (0, 1 or 2). FairTMJ also
uses column 5 for binary sex (0 or 1). Other columns are ignored. Baseline
loading requires only the first four columns and does not parse sex.

Each joint has three Closed-PD and three Open-PD JPEG slices under
<data_folder>/<patient_id>/<patient_id>_<side>/. Patient IDs and joint sides
are metadata for grouping and export; the diagnosis model receives images.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_records(path, *, include_sex=True):
    """Read required annotation fields and validate joint and patient metadata."""
    records, seen, patient_sex = [], set(), {}
    with open(path, encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.strip().split(";")]
            required_columns = 6 if include_sex else 4
            if len(fields) < required_columns:
                raise ValueError(f"{path}:{line_number}: expected at least {required_columns} columns")
            patient, side, label = fields[0], fields[1], int(fields[3])
            if not patient or not side or label not in (0, 1, 2):
                raise ValueError(f"{path}:{line_number}: invalid ID, side or ADD label")
            if (patient, side) in seen:
                raise ValueError(f"{path}:{line_number}: duplicate patient/side {(patient, side)}")
            seen.add((patient, side))
            record = dict(patient_id=patient, side=side, label=label)
            if include_sex:
                sex = int(fields[5])
                if sex not in (0, 1):
                    raise ValueError(f"{path}:{line_number}: sex must be coded as 0 or 1")
                if patient in patient_sex and patient_sex[patient] != sex:
                    raise ValueError(f"{path}:{line_number}: inconsistent sex within patient {patient}")
                patient_sex[patient] = sex
                record["sex"] = sex
            records.append(record)
    if not records:
        raise ValueError(f"Empty annotation file: {path}")
    return records


class ADDMRIDataset(Dataset):
    """Load six MRI slices with ADD labels and optional sex metadata.

    Provide either an annotation ``file`` or parsed ``records`` from
    ``load_records``. ``include_sex=False`` returns five fields: closed-mouth
    images, open-mouth images, label, patient ID and side. With sex enabled,
    the sex code is inserted after the label, giving six fields.

    The default transform resizes to 512 x 512, crops the central 256 x 256
    region and converts RGB images to tensors. A custom callable can replace
    this transform. Paths and study-specific splits are supplied by callers.
    """

    def __init__(self, data_folder, file=None, transform=None, *, records=None, include_sex=True):
        if (file is None) == (records is None):
            raise ValueError("Provide exactly one of file or records")
        self.data_folder = Path(data_folder)
        self.include_sex = include_sex
        source_records = load_records(file, include_sex=include_sex) if records is None else records
        # Keep only fields used by this mode, even when callers pass richer records.
        fields = ("patient_id", "side", "label", "sex") if include_sex else ("patient_id", "side", "label")
        self.records = [{key: record[key] for key in fields} for record in source_records]
        if not self.records:
            raise ValueError("The dataset must contain at least one joint")
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def _load_images(self, folder, prefix):
        images = []
        for index in range(1, 4):
            with Image.open(folder / f"{prefix}{index}.jpg") as image:
                images.append(self.transform(image.convert("RGB")))
        return images

    def __getitem__(self, index):
        record = self.records[index]
        folder = self.data_folder / record["patient_id"] / f'{record["patient_id"]}_{record["side"]}'
        closed = self._load_images(folder, "Closed-PD")
        opened = self._load_images(folder, "Open-PD")
        if self.include_sex:
            return closed, opened, record["label"], record["sex"], record["patient_id"], record["side"]
        return closed, opened, record["label"], record["patient_id"], record["side"]
