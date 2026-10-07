// Keep model file names here until the classic builders expose presets.
pub(crate) struct ModelFiles {
    pub(crate) det: &'static str,
    pub(crate) rec: &'static str,
    pub(crate) dict: &'static str,
    pub(crate) layout: &'static str,
    pub(crate) table_dict: &'static str,
    pub(crate) table_classifier: &'static str,
    pub(crate) wired_structure: &'static str,
    pub(crate) wireless_structure: &'static str,
    pub(crate) wired_cells: &'static str,
}

pub(crate) const MODELS: ModelFiles = ModelFiles {
    det: "pp-ocrv6_tiny_det.onnx",
    rec: "pp-ocrv6_tiny_rec.onnx",
    dict: "ppocrv6_tiny_dict.txt",
    layout: "pp-doclayoutv3.onnx",
    table_dict: "table_structure_dict_ch.txt",
    table_classifier: "pp-lcnet_x1_0_table_cls.onnx",
    wired_structure: "slanext_wired.onnx",
    wireless_structure: "slanet_plus.onnx",
    wired_cells: "rt-detr-l_wired_table_cell_det.onnx",
};
