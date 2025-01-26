from enum import Enum


# this type is about the type of tensor that a board is turned into
class InternalTensorRepresentationType(str, Enum):
    NOBUG364 = "364_no_bug"
    BUG364 = "364_bug"
    NO = "no"


# this type is more about the technical implementation of internal representation within a node
class TensorRepresentationType(str, Enum):
    NOBUG364 = "364_no_bug"
    BUG364 = "364_bug"


def get_default_internal_representation(
    tensor_representation_type: TensorRepresentationType,
) -> InternalTensorRepresentationType:
    match tensor_representation_type:
        case TensorRepresentationType.NOBUG364:
            return InternalTensorRepresentationType.NOBUG364
        case TensorRepresentationType.BUG364:
            return InternalTensorRepresentationType.BUG364


compatibilities: dict[
    TensorRepresentationType, list[InternalTensorRepresentationType]
] = {
    TensorRepresentationType.BUG364: [InternalTensorRepresentationType.BUG364],
    TensorRepresentationType.NOBUG364: [InternalTensorRepresentationType.NOBUG364],
}


def assert_compatibilities_representation_type(
    tensor_representation_type: TensorRepresentationType,
    internal_tensor_representation_type: InternalTensorRepresentationType,
) -> None:
    assert (
        internal_tensor_representation_type
        in compatibilities[tensor_representation_type]
    )
