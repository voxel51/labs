import os

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
from fiftyone.core.expressions import ViewField as F

os.environ["VFF_EXP_ANNOTATION"] = "1"
print("\n--------------------------------")
print("Make sure your fiftyone installation is on the latest `develop`")
print("--------------------------------\n")


dataset = foz.load_zoo_dataset(
    "https://github.com/voxel51/davis-2017",
    split="validation",
    format="image",
)
dataset.persistent = True

SELECT_SEQUENCES = ["dogs-jump"]
view = dataset.match_tags(SELECT_SEQUENCES)
view = view.match(F("frame_number").to_int() < 9)

session = fo.launch_app(view)

print("\n--------------------------------")
print("Annotate some frames and propagate with the operator")
print("--------------------------------\n")

input("Press Enter to close the app...")
session.close()
