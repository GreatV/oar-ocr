//! Static registry of model files hosted on ModelScope under
//! [`greatv/oar-ocr`](https://www.modelscope.cn/models/greatv/oar-ocr).
//!
//! The entries are sorted by name to allow a binary search lookup.
//! This file is intended to be regenerated as new models are uploaded.

use super::Entry;

/// ModelScope owner/name of the repository hosting the registered files.
pub const MODELSCOPE_REPO: &str = "greatv/oar-ocr";

/// Default revision (branch/tag) to fetch from ModelScope.
pub const DEFAULT_REVISION: &str = "master";

/// Sorted-by-name registry of files mirrored on ModelScope.
///
/// Adding an entry: upload the file with `modelscope upload greatv/oar-ocr`,
/// compute its SHA-256 (`sha256sum`) and byte size (`stat -c %s`), and insert
/// in name order. Keep the slice sorted so [`super::find`] can binary-search.
#[rustfmt::skip]
pub static REGISTRY: &[Entry] = &[
    Entry { name: "arabic_pp-ocrv3_mobile_rec.onnx", sha256: "7012a52bfe7ed9910bef1c74e295d8c3456175aa9b4e9015271892ced559687a", size: 8995821 },
    Entry { name: "arabic_pp-ocrv5_mobile_rec.onnx", sha256: "2768206d9a0ce48eba45b59619184e18161dde8f44115f029920ca17a9dc0384", size: 8026538 },
    Entry { name: "ch_repsvtr_rec.onnx", sha256: "6e7d40f0e3c4c16443c9efe7265bb9788ee3461fd6f26b41990524fbaae7ec5d", size: 25371710 },
    Entry { name: "ch_svtrv2_rec.onnx", sha256: "3fadaeecebd49d4df4f96155875be393e66161befc26258d3e62ee9968efd648", size: 84196641 },
    Entry { name: "chinese_cht_pp-ocrv3_mobile_rec.onnx", sha256: "535ec8a5d0207f34b4cc6cca28c0903991df0ec6ecb8bd7eea2f16f4468afc2b", size: 11143425 },
    Entry { name: "cyrillic_pp-ocrv3_mobile_rec.onnx", sha256: "6ab2b46cee27755f82cacd86a73706f00146f1938aa5c74549a4fb2d1f94ae9c", size: 8996341 },
    Entry { name: "cyrillic_pp-ocrv5_mobile_rec.onnx", sha256: "a18d96d7c8d73d90f2ed056549caa1de3a8e6cb744cccba16cd593ea8cd2d569", size: 8076390 },
    Entry { name: "devanagari_pp-ocrv3_mobile_rec.onnx", sha256: "97bc713646ae30442d536b47c3f0d65ad249dbc8006a7beff66825f7df691405", size: 8997381 },
    Entry { name: "devanagari_pp-ocrv5_mobile_rec.onnx", sha256: "b3d50774dfbec6ae02249ff79a925431a4381c8c6f86d342ff6e7b63e5fefa77", size: 7939902 },
    Entry { name: "el_pp-ocrv5_mobile_rec.onnx", sha256: "5a4a020e48e8783e035e1af135423c2161a363acab9ef16e48238c3d181f0f71", size: 7836326 },
    Entry { name: "en_pp-ocrv3_mobile_rec.onnx", sha256: "dcb188df82c426f251283fd4ef4ea57c039520ce6ff8355ba5aa9b2535073c33", size: 8978654 },
    Entry { name: "en_pp-ocrv4_mobile_rec.onnx", sha256: "40c07c5e431a4c59d7b5a1fefdba2fddb962c939d626c4dbf1d32965ab533431", size: 7710963 },
    Entry { name: "en_pp-ocrv5_mobile_rec.onnx", sha256: "8307465d3c9ef2ba4055c3bd0be55aafe11f518630212b7598b70ccb376028ac", size: 7876014 },
    Entry { name: "eslav_pp-ocrv5_mobile_rec.onnx", sha256: "36a66a68097e88b103e0f60f489e88c7239d3ea79d96fbac2d80ac9d134944cd", size: 7915218 },
    Entry { name: "japan_pp-ocrv3_mobile_rec.onnx", sha256: "da59d4dbb6786a92a3823c32b3f48179d18e29903d61462842aad9eb422a77a1", size: 10097703 },
    Entry { name: "ka_pp-ocrv3_mobile_rec.onnx", sha256: "e9592deee670c7ae3b21a7f6ad5d73f080628a30d8f00d26c668cfa3b493de04", size: 8993741 },
    Entry { name: "korean_pp-ocrv3_mobile_rec.onnx", sha256: "d30dbf20502044dc0e697f047564ac015e7083e4766fdc9d0fcd225aa2b0f20d", size: 9912841 },
    Entry { name: "korean_pp-ocrv5_mobile_rec.onnx", sha256: "2d7ed96308065a86103325d22af07a88c4d06afc009f21602a4882342c0cc054", size: 13446374 },
    Entry { name: "latex_ocr_rec.onnx", sha256: "b4714a09ab4b5049ef5d404b4f8212e17ee03d634756ad6ba3ad380e91745613", size: 102499156 },
    Entry { name: "latin_pp-ocrv3_mobile_rec.onnx", sha256: "e73a1fc3853b36fe99d7990858bbc9630346706db2e08738ffd352cf789de7ef", size: 9002061 },
    Entry { name: "latin_pp-ocrv5_mobile_rec.onnx", sha256: "e3a6bfeea1c8a01d6fccfd480a0bd363fd907f8c65931e228bb2736f5c3e142f", size: 8069614 },
    Entry { name: "p2o_pp-lcnet_x0_25_textline_ori.onnx", sha256: "44fdaeabcd95861fcdf8a31f8ecebf885f72ce50dac94989603a3bc60eacde54", size: 1000746 },
    Entry { name: "picodet-l_layout_17cls.onnx", sha256: "bf16cde3c9d0fe160ef74d7f9143f67f4c85fb0a6afe3923942ebe8e8854e734", size: 23481268 },
    Entry { name: "picodet-l_layout_3cls.onnx", sha256: "2112c55bb86aa59dc6bf4d71b75266e5b07fdb663eb295967086cd657d870001", size: 23445204 },
    Entry { name: "picodet-s_layout_17cls.onnx", sha256: "94da097a087d039a87575a3f1172b19265a74b40781aaf6727117a2205b4cb6d", size: 4905603 },
    Entry { name: "picodet-s_layout_3cls.onnx", sha256: "77895ad1aff11d7b4fd94d58454db0a848b90beabfac34a74feaad3956a277ca", size: 4883874 },
    Entry { name: "picodet_layout_1x.onnx", sha256: "5a95d6a17380cc5b146f515548679046708fd18d8caa1cafecd98a80b6252523", size: 7522719 },
    Entry { name: "picodet_layout_1x_table.onnx", sha256: "e62882aa0eedd7aa417e7a1ef0042f1a106a2492f32c2fed492e9b3145c2d1ba", size: 7514462 },
    Entry { name: "pp-docblocklayout.onnx", sha256: "4f2b7465a9ca1e8519848573544e1ee108ead67f8b35958b77a293b02eca44cd", size: 129331821 },
    Entry { name: "pp-doclayout-l.onnx", sha256: "094fef666d9785d001238d0b93a88c2c365b059d211d7e1e494ccd2419f3b3c1", size: 129377057 },
    Entry { name: "pp-doclayout-m.onnx", sha256: "8e458bfc919bbf7a35be9802485b5cd30151cb356364cfad09911d2ee1fc1f76", size: 23496727 },
    Entry { name: "pp-doclayout-s.onnx", sha256: "c2336493a0a13cd9b9b457ca68aea370b327c362a4a7da4917c2bba96029bceb", size: 4914918 },
    Entry { name: "pp-doclayout_plus-l.onnx", sha256: "b06cedc7ab3cca7da4ed66cf16024732149d2c29e6adcbfc69b9bb6ef94b4a48", size: 129714689 },
    Entry { name: "pp-doclayoutv2.onnx", sha256: "a325532df1c7530538ef4e8254695c091adc6afd3366c0851425491f0816d1d1", size: 213969379 },
    Entry { name: "pp-doclayoutv3.onnx", sha256: "1a7ec3812d239ad14debb87e38273012558fea27ac10b44b61260e3a88358e39", size: 129955811 },
    Entry { name: "pp-formulanet-l.onnx", sha256: "e408f1d4e6d67c694a2c8f75dfd17d2e2668b1191c3c68d029d445b133be4bb5", size: 730379948 },
    Entry { name: "pp-formulanet-s.onnx", sha256: "0ee32c7bfbd9e586364f89f71860476ccb5334e35674a61f3df5e0553d6a6dcc", size: 231878904 },
    Entry { name: "pp-formulanet-tokenizer.json", sha256: "2811d82701ec97c192fa256aa2b4516929373870ae660326cc5b1dc879b95ff2", size: 2140014 },
    Entry { name: "pp-formulanet_plus-l.onnx", sha256: "b4924d69c731365048de3d11a5d1829f3dfd8b98b4dbfd82437f934c2611934f", size: 733525676 },
    Entry { name: "pp-formulanet_plus-m.onnx", sha256: "9e3539c2b4eeed28f2d35e342fd5bb0bdaa7f6034a475fc7e890c92780910618", size: 592372919 },
    Entry { name: "pp-formulanet_plus-s.onnx", sha256: "449d205c8fb2fe0a9b134a5e4a0f2421c2e7812fd902ea67dfda4e9ef4588978", size: 231878904 },
    Entry { name: "pp-lcnet_x0_25_textline_ori.onnx", sha256: "fb402220f39b183d64a68cd48d4bd53267a21354b6fc39370c2a83fbdad85b10", size: 1018629 },
    Entry { name: "pp-lcnet_x1_0_doc_ori.onnx", sha256: "bbcd6c2b43ab15d2e605455aef2cd280ab87b570824522d24a69e9298875a1ac", size: 6787248 },
    Entry { name: "pp-lcnet_x1_0_table_cls.onnx", sha256: "61ed75151cadba903ec5182f1ffc59e961e52de501c61c5ffeb466346fc65040", size: 6776998 },
    Entry { name: "pp-lcnet_x1_0_textline_ori.onnx", sha256: "6b02efabbedd6be69e3de4c86b8dceed2d7329e75c12a796e6717bfb0d646950", size: 6776997 },
    Entry { name: "pp-ocrv3_mobile_rec.onnx", sha256: "8febeeba4792aed934be20ffffa6f050d717da059084cc294d4a88ee35130599", size: 10675943 },
    Entry { name: "pp-ocrv4_mobile_det.onnx", sha256: "ab2a50dcd2c340852f2d0fbfa547d5eec79a0d04a774eb0b622d96d0d9d2ceeb", size: 4826518 },
    Entry { name: "pp-ocrv4_mobile_rec.onnx", sha256: "5d54b59bac0f49d4561f0462630d8a6822b5b495db064f58096fd3d2392fbc4e", size: 10870526 },
    Entry { name: "pp-ocrv4_mobile_seal_det.onnx", sha256: "e6109a1022b5ebf0822fc00646ef2398a7ef387390ca5c978de79352b1314204", size: 4826518 },
    Entry { name: "pp-ocrv4_server_det.onnx", sha256: "5b676249ca4d1653675b249f134cb483ada721ac70d8013ddab09db7bcf26c1f", size: 113442336 },
    Entry { name: "pp-ocrv4_server_rec.onnx", sha256: "70939bcaabb8700dd9627ab7a38acbbbb8eea589cf27aef565f2343921a502c8", size: 90538610 },
    Entry { name: "pp-ocrv4_server_rec_doc.onnx", sha256: "1c64b0b01d5e03b931608ee366efeca868a6fd1b8015bd8f4f9a1fff43708ae3", size: 94897514 },
    Entry { name: "pp-ocrv4_server_seal_det.onnx", sha256: "8fc8b257e3841144c23b2d75b35cd95d82abefa343b6ac16f615f96c848e2357", size: 113442336 },
    Entry { name: "pp-ocrv5_mobile_det.onnx", sha256: "1eb7b4f7ab657ebd1c66d5f79bca7497f29768a2e3c15e52daecbba1a8e4a039", size: 4826518 },
    Entry { name: "pp-ocrv5_mobile_rec.onnx", sha256: "243a0f06d826761323e9045e9b113ab2c191c3aa50565585e628300b8eda0224", size: 16562373 },
    Entry { name: "pp-ocrv5_server_det.onnx", sha256: "9a910baffbefb807ff2f7bfaa72910e3e470bd17014d798386d87bb46f442839", size: 88116836 },
    Entry { name: "pp-ocrv5_server_rec.onnx", sha256: "4bfffad2c62eb1340250455856978fb9fb19cb4776b264ae3c2f91c35fbb40b4", size: 84502992 },
    Entry { name: "ppocr_keys_v1.txt", sha256: "a1c84d9bdb9ab29043c58896224d32941783eb821629618416dcb08f12886492", size: 26250 },
    Entry { name: "ppocrv4_doc_dict.txt", sha256: "a5bc3887c43c901e5a3f97b13ffadf1c5754ede7cc8c9f5abe22e875a7c48372", size: 62346 },
    Entry { name: "ppocrv5_arabic_dict.txt", sha256: "7f92f7dbb9b75a4787a83bfb4f6d14a8ab515525130c9d40a9036f61cf6999e9", size: 2369 },
    Entry { name: "ppocrv5_cyrillic_dict.txt", sha256: "db40aa52ceb112055be80c694afdf655d5d2c4f7873704524cc16a447ca913ba", size: 2781 },
    Entry { name: "ppocrv5_devanagari_dict.txt", sha256: "09c7440bfc5477e5c41052304b6b185aff8c4a5e8b2b4c23c1c706f6fe1ee9fc", size: 1943 },
    Entry { name: "ppocrv5_dict.txt", sha256: "d1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b", size: 74012 },
    Entry { name: "ppocrv5_el_dict.txt", sha256: "31defc62c0c3ad3674a82da6192226a2ba98ef4ff014a7045cb88d59f9c3de31", size: 1103 },
    Entry { name: "ppocrv5_en_dict.txt", sha256: "e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6", size: 1416 },
    Entry { name: "ppocrv5_eslav_dict.txt", sha256: "3e95f1581557162870cacdba5af91a4c6be2890710d395b0c3c7578e7ee5e6eb", size: 1663 },
    Entry { name: "ppocrv5_korean_dict.txt", sha256: "a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7", size: 47451 },
    Entry { name: "ppocrv5_latin_dict.txt", sha256: "ccbcc45730b3fbbd9050c5bc74db6a99067141ef1035e3d14889a84a6b9b1aff", size: 2616 },
    Entry { name: "ppocrv5_ta_dict.txt", sha256: "85b541352ae18dc6ba6d47152d8bf8adff6b0266e605d2eef2990c1bf466117b", size: 1723 },
    Entry { name: "ppocrv5_te_dict.txt", sha256: "42f83f5d3fdb50778e4fa5b66c58d99a59ab7792151c5e74f34b8ffd7b61c9d6", size: 1831 },
    Entry { name: "ppocrv5_th_dict.txt", sha256: "57f5406f94bb6688fb7077f7be65f08bbd71cecf48c01ea26c522cb5c4836b7a", size: 1767 },
    Entry { name: "rt-detr-h_layout_17cls.onnx", sha256: "079173c137540a2a56598d872e408646af12aa537140c1dc246592af4e7f9b95", size: 492056102 },
    Entry { name: "rt-detr-h_layout_3cls.onnx", sha256: "bce52ce49762f77213b2dd40ab5901c504e809880cc3e684f97e472f2a3303aa", size: 492027314 },
    Entry { name: "rt-detr-l_wired_table_cell_det.onnx", sha256: "238dfece5c48d926a3ebac07341eb197f35038f5f5ca79dc6f75fa9686853f6d", size: 129331821 },
    Entry { name: "rt-detr-l_wireless_table_cell_det.onnx", sha256: "3b373ba8467403956e2f043bfc00fba8a147fcb18c6988b898776d5cd523f520", size: 129331821 },
    Entry { name: "slanet.onnx", sha256: "ebb506f2af6ba26502bb857b6f82a06af12c5231a1c52146a473b2c90205df3b", size: 7782138 },
    Entry { name: "slanet_plus.onnx", sha256: "3a96a71719247c5d94992fca31266b598c54740388de371f0c75077e2a9e0b55", size: 7782138 },
    Entry { name: "slanet_plus_v2.onnx", sha256: "e0bff8da087f9b83629f1e1a6e0f8252fc2de85a7d80415b3510fc521338da3d", size: 7781255 },
    Entry { name: "slanext_wired.onnx", sha256: "0d1efd752685f42271326eeca93f321fc6ba6d6f75ff491f31f40556dcecc4af", size: 367743373 },
    Entry { name: "slanext_wireless.onnx", sha256: "9bc8f145da44766c11acef4e436a58da8fb192ef50dc7ffc2d7fcdf82ae66419", size: 367743373 },
    Entry { name: "ta_pp-ocrv3_mobile_rec.onnx", sha256: "de98658698cf72be6f299f04ed78032489ab502553dab31b13f49f22dab2d62f", size: 8987241 },
    Entry { name: "ta_pp-ocrv5_mobile_rec.onnx", sha256: "508d07ac0e1806a8b8857ebf20bd8837d68d962c3fdba030cf0022238ba819b8", size: 7913282 },
    Entry { name: "table_structure_dict_ch.txt", sha256: "68d344a84b726e043f390122240ff2b2ced2949b2a80ce9b61ae955054d190ef", size: 578 },
    Entry { name: "te_pp-ocrv3_mobile_rec.onnx", sha256: "5a8ba806f4fecac40bbaa468e1885530eaea483f9c63b01dbe93acebc342caa0", size: 8993221 },
    Entry { name: "te_pp-ocrv5_mobile_rec.onnx", sha256: "0957fc03425324d30afcd4342a5708687fd0c885a5c0987cae418e96dd761a60", size: 7926350 },
    Entry { name: "th_pp-ocrv5_mobile_rec.onnx", sha256: "5f6ee21242691681261fee01bc39867da9cc8ff9b889f2f048b3cb7f74380217", size: 7918606 },
    Entry { name: "unimernet.onnx", sha256: "1d64fafa0161f153dafe40823e97c4b05103030509dd6b286d7c8d4a11b068ab", size: 1842024100 },
    Entry { name: "unimernet_tokenizer.json", sha256: "2811d82701ec97c192fa256aa2b4516929373870ae660326cc5b1dc879b95ff2", size: 2140014 },
    Entry { name: "unimernet_tokenizer_config.json", sha256: "fd4d94f8b9dbb7deeb3a3ef084ca0e16c43d45774a180730b7e9cfd6359a074b", size: 4491 },
    Entry { name: "uvdoc.onnx", sha256: "1092557894d49644e7858b293df6cb9c873d53e51319b91a8614ca9c71686dc0", size: 31684150 },
];

#[cfg(test)]
mod tests {
    use super::REGISTRY;

    #[test]
    fn registry_is_sorted_and_unique() {
        for pair in REGISTRY.windows(2) {
            assert!(
                pair[0].name < pair[1].name,
                "registry must be sorted and unique, but `{}` ≥ `{}`",
                pair[0].name,
                pair[1].name,
            );
        }
    }

    #[test]
    fn registry_hashes_are_64_lowercase_hex() {
        for entry in REGISTRY {
            assert_eq!(
                entry.sha256.len(),
                64,
                "{} has non-64-char sha256",
                entry.name
            );
            assert!(
                entry
                    .sha256
                    .chars()
                    .all(|c| matches!(c, '0'..='9' | 'a'..='f')),
                "{} sha256 must be lowercase hex",
                entry.name
            );
        }
    }
}
