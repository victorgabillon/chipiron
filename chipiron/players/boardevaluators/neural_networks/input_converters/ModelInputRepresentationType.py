"""Get the default internal tensor representation for a given model input representation type."""

from enum import Enum


# this type is more about the technical implementation of internal representation within a node
class InternalTensorRepresentationType(str, Enum):
    """Defines the types of internal tensor representations."""

    NOBUG364 = "364_no_bug"
    BUG364 = "364_bug"
    NO = "no"


# this type is about the type of representation input fed to NN models
class ModelInputRepresentationType(str, Enum):
    """Defines the types of model input representations."""

    NOBUG364 = "364_no_bug"
    BUG364 = "364_bug"
    PIECE_MAP = "piece_map"
    PIECE_DIFFERENCE = "pieces_difference"
    BOARD_PIECES_TWO_SIDES = "board_pieces_two_sides"


def get_default_internal_representation(
    model_input_representation_type: ModelInputRepresentationType,
) -> InternalTensorRepresentationType:
    """Get the default internal tensor representation for a given model input representation type.

    Args:
        model_input_representation_type (ModelInputRepresentationType): The model input representation type.

    Returns:
        InternalTensorRepresentationType: The default internal tensor representation type.
    """
    match model_input_representation_type:
        case ModelInputRepresentationType.NOBUG364:
            return InternalTensorRepresentationType.NOBUG364
        case ModelInputRepresentationType.BUG364:
            return InternalTensorRepresentationType.BUG364
        case _:
            return InternalTensorRepresentationType.NO


compatibilities: dict[
    ModelInputRepresentationType, list[InternalTensorRepresentationType]
] = {
    ModelInputRepresentationType.BUG364: [InternalTensorRepresentationType.BUG364],
    ModelInputRepresentationType.NOBUG364: [InternalTensorRepresentationType.NOBUG364],
}


def assert_compatibilities_representation_type(
    model_input_representation_type: ModelInputRepresentationType,
    internal_tensor_representation_type: InternalTensorRepresentationType,
) -> None:
    """
    Assert that the internal tensor representation type is compatible with the model input representation type.

    Args:
        model_input_representation_type (ModelInputRepresentationType): The model input representation type.
        internal_tensor_representation_type (InternalTensorRepresentationType): The internal tensor representation type.

    Raises:
        AssertionError: If the internal tensor representation type is not compatible with the model input representation type.
    """
    assert (
        internal_tensor_representation_type
        in compatibilities[model_input_representation_type]
    )
