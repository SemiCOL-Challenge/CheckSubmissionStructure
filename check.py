import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, BadZipFile

import PIL.Image
import numpy as np
from sklearn.metrics import roc_auc_score

for archive_path in sys.argv[1:]:
    print(f"\nChecking archive: {archive_path}")
    archive_path = Path(archive_path)

    # Check that the archive can be found on the hard-drive
    if not archive_path.is_file():
        print(f"\tERROR:  \tThe archive could not be found. Please check the path that you supplied.")
        continue

    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        # Try to unzip the archive in a temporary directory
        try:
            with ZipFile(archive_path, "r") as f:
                f.extractall(tempdir)
        except BadZipFile:
            print("\tERROR:  \tThis ZIP file is corrupted and cannot be opened.")
            continue

        # ==============================================================================================================
        # Check classification.json
        skip_classification_check = False
        any_classification_error = False
        path_classification = tempdir / "classification.json"

        if not path_classification.is_file():
            print("\tERROR:  \tclassification.json is missing.")
            any_classification_error = True
        elif path_classification.is_dir():
            print("\tERROR:  \tclassification.json is a directory.")
            any_classification_error = True
        else:
            with open(path_classification, "r") as f:
                # Try to read the file
                try:
                    predictions = json.load(f)
                except UnicodeDecodeError:
                    print("\tERROR:  \tclassification.json contains invalid unicode characters.")
                    skip_classification_check = True
                    any_classification_error = True
                except json.JSONDecodeError:
                    print("\tERROR:  \tclassification.json is not a valid JSON file."
                          "http://json.parser.online.fr/ can tell you what the issue is...")
                    skip_classification_check = True
                    any_classification_error = True

                if not skip_classification_check:
                    # Check that all keys are there / are expected
                    # noinspection SpellCheckingInspection
                    expected_keys = {'FoSJZXMf', 'QWbE525I', 'rXnHCUvh', 'QgiI3ZjZ', 'mEbHuWpr', 'QALoeYtI', 'ekSUm2cR',
                                     'IWHbSh5u', 'nDEUHXwj', 'JMo8YkLs', 'DsRY1rxV', 'eRLJEqUc', 'I5eHKnl9', 'TE4z9RQi',
                                     'pijPwLr3', 'JEKP6PPk', 'VkwcFVCj', 'gXwu1b4V', 'OuM2aSiS', '2pT4GFzn', '9jNhMVzm',
                                     'Jf22jnO2', 'DZVCFC0G', 'yUTVZuRr', 'XS7iESy5', 'ABLVABMT', 'usRRkq2C', 'J9cGjQjN',
                                     'DokmCXM1', 'tUw6bIr8', 'VcspD4KF', 'ft5ALWqc', '4Hjkjs0i', '9bwsptDH', 'AlTjYbsA',
                                     'tJgvMquY', 'x2cYOIty', 'NUYZ3vYd', 'F62qzus2', 'BHTUesDo'}
                    keys_found = set(predictions.keys())

                    missing_keys = expected_keys.difference(keys_found)
                    if missing_keys:
                        print(f"\tERROR:  \tclassification.json is missing the following keys {list(missing_keys)}")
                        any_classification_error = True

                    unexpected_keys = keys_found.difference(expected_keys)
                    if unexpected_keys:
                        print(f"\tWARNING:\tclassification.json contains the following unexpected keys "
                              f"{list(unexpected_keys)}")
                        any_classification_error = True

                    # Check that all values are float in [0, 1]
                    for key in keys_found:
                        value = predictions[key]
                        if not isinstance(value, float) and not isinstance(value, int):
                            print(f"\tERROR:  \tIn classification.json, "
                                  f"key {key} has a value of type {type(predictions[key])} (instead of float).")
                            any_classification_error = True
                        elif value < 0 or value > 1:
                            print(f"\tWARNING:\tIn classification.json, "
                                  f"key {key} has a float value ({value}) not in range [0, 1].")
                            any_classification_error = True

                    # Double check that the auc can be computed with dummy data
                    if len(predictions) > 2 and not any_classification_error:
                        try:
                            # Need at least 2 different values to compute the AUC
                            dummy_gt = [1] + [0] * (len(predictions) - 1)
                            roc_auc_score(dummy_gt, list(predictions.values()))
                        except Exception as e:
                            print(f"\tERROR:  \tUnexpected error when computing the AUC {e}")
                            any_classification_error = True

        if not any_classification_error:
            print("\tSUCCESS:\tclassification.json is valid!")

        # ==============================================================================================================
        # Check segmentations
        cases = 6
        any_segmentation_error = False

        with open("expected_patches.json", "r") as f:
            all_expected_patches = json.load(f)

        expected_classes = list(range(1, 11))
        expected_shape = 3000, 3000

        for case in range(1, 1 + cases):
            case_str = f"wns_case_{case:02d}"
            case_path = tempdir / case_str
            # Check case folders
            if not case_path.is_dir():
                print(f"\tWARNING:\tFolder for case '{case_str}' is missing.")
                any_segmentation_error = True
            elif case_path.is_file():
                print(f"\tERROR:  \tFolder for case '{case_str}' is a file.")
                any_segmentation_error = True
            else:
                # Check that all patches are there / are expected
                expected_patches = set(all_expected_patches[str(case)])
                patches_found = set(p.name for p in case_path.glob("*.png"))

                missing_patches = expected_patches.difference(patches_found)
                if missing_patches:
                    print(f"\tERROR:  \t{case_str} is missing the following patches [{list(missing_patches)}]")
                    any_segmentation_error = True

                unexpected_patches = patches_found.difference(expected_patches)
                if unexpected_patches:
                    print(f"\tWARNING:\t{case_str} contains the following unexpected patches "
                          f"{list(missing_patches)}")
                    any_segmentation_error = True

                # Check each patch
                for patch in case_path.glob("*.png"):
                    try:
                        # noinspection PyTypeChecker
                        array = np.asarray(PIL.Image.open(patch))

                        if array.dtype != np.uint8:
                            print(f"\tERROR:\t{patch.name} has dtype {array.dtype} (instead of uint8).")
                            any_segmentation_error = True
                        if array.shape != expected_shape:
                            print(f"\tERROR:  \t{patch.name} has shape {array.shape} (instead of {expected_shape}).")
                            any_segmentation_error = True
                        unexpected_classes = set(np.unique(array)).difference(expected_classes)
                        if unexpected_classes:
                            print(f"\tWARNING:\t{patch.name} contains unexpected classes {list(unexpected_classes)}")
                            any_segmentation_error = True
                    except (PermissionError, PIL.UnidentifiedImageError):
                        print(f"\tERROR:  \t{patch.name} is not a valid PNG file.")
                        any_segmentation_error = True

        if not any_segmentation_error:
            print("\tSUCCESS:\tNo errors in the segmentations files!")

        if not any_classification_error and not any_segmentation_error:
            print("\tSUCCESS:\tThe submission is completely valid!")
