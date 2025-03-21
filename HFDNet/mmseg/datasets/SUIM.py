from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class SUIM(BaseSegDataset):
    METAINFO = dict(
        classes=('Human divers', 'Wrecks and ruins', 'Robots', 'Reefs and invertebrates',
                 'Fish and vertebrates'),
        palette=[[0,0,255], [0,255,255],
                 [255,0,0], [255,0,255], [255,255,0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
