"""CTC emissions to frame alignments, without acoustic model dependencies."""

from .alignment import AlignmentBatch, align, emission_validity, load_extension, validate_inputs

__all__ = ["AlignmentBatch", "align", "emission_validity", "load_extension", "validate_inputs"]
