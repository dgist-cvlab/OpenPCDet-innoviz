from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_head_simple_yesz import PointHeadSimpleYesz
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .center_head_noz import CenterHeadNoz
from .trans_head import TransHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadSimpleYesz': PointHeadSimpleYesz,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'CenterHeadNoz': CenterHeadNoz,
    'TransHead' : TransHead,
}
