from packer import (
    IrregularPacker,
    IrregularPackerStrict,
    IrregularPackerStrictGBFLS,
    IrregularPackerIntensification,
    IrregularPackerStrictIntensification,
    IrregularPackerLSIntensification,
    IrregularPackerStrictLSIntensification,
    IrregularPackerPSOLS,
    IrregularPackerPSO,
    IrregularPackerGBFLS,
    IrregularPackerStrictPSO,
    IrregularPackerBF,
    IrregularPackerStrictPSOLS,
    IrregularPackerStrictGridBF,
    IrregularPackerGridBF,
    IrregularPackerStrictBF,
    CirclePackerBeeIntensification,
    CirclePackerBeeStrictIntensification,
    CirclePacker,
    CirclePackerStrict,
    CirclePackerBeeStrictLS,
    CirclePackerBeeLS,
    CirclePackerBeeLSIntensification,
    CirclePackerBeeStrictLSIntensification,
    CirclePackerBeeStrict,
    CirclePackerPSOLS,
    CirclePackerPSO,
    CirclePackerStrictPSO,
    CirclePackerBee,
    CirclePackerStrictPSOLS
)

CONSTRUCTIVE_MODELS = [
    # IrregularPacker,
    # IrregularPackerStrict,
    # IrregularPackerBF,
    # IrregularPackerStrictBF,
    IrregularPackerGridBF,
    IrregularPackerStrictGridBF,
    # CirclePacker,
    # CirclePackerStrict,
    CirclePackerBee,
    CirclePackerBeeStrict,
]

INTENSIFICATION_MODELS = [
    IrregularPackerIntensification,
    IrregularPackerStrictIntensification,
    IrregularPackerLSIntensification,
    IrregularPackerStrictLSIntensification,
    IrregularPackerGBFLS,
    IrregularPackerStrictGBFLS,
    CirclePackerBeeIntensification,
    CirclePackerBeeStrictIntensification,
    CirclePackerBeeLS,
    CirclePackerBeeStrictLS,
    CirclePackerBeeLSIntensification,
    CirclePackerBeeStrictLSIntensification,
]

PSO_MODELS = [
    IrregularPackerPSO,
    IrregularPackerStrictPSO,
    CirclePackerPSO,
    CirclePackerStrictPSO,
]

COMBINED_MODELS = [
    IrregularPackerPSOLS,
    IrregularPackerStrictPSOLS,
    CirclePackerPSOLS,
    CirclePackerStrictPSOLS,
]

ALL_MODELS = COMBINED_MODELS + INTENSIFICATION_MODELS + PSO_MODELS + COMBINED_MODELS
