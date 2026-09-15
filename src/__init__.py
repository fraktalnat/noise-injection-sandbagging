"""
Source code for the project.
"""

import logging

# Library code logs via `logging` rather than `print`. Following the standard
# library convention, the package attaches a NullHandler and leaves handler
# configuration to the entry-point script. To see the `verbose=True` output from
# `eval_model_on_dataset_batches`, call `logging.basicConfig(level=logging.INFO)`
# in the script before running an evaluation.
logging.getLogger(__name__).addHandler(logging.NullHandler())
