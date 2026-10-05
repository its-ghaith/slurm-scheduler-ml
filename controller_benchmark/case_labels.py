from __future__ import annotations


# Stable appendix labels for the 48-case thesis campaign.  The final character
# identifies Scratch (S) or Pretrained (P); every code is exactly three chars.
CASE_LABELS: dict[str, tuple[str, str, int]] = {
    "carpk-aerial-vehicle": ("CRS", "CARPK — Scratch", 1),
    "vedai-aerial-vehicle": ("VDS", "VEDAI 512 — Scratch", 2),
    "visdrone-aerial-vehicle": ("VSS", "VisDrone2019-DET Vehicles — Scratch", 3),
    "cifar10-classification": ("C1S", "CIFAR-10 — Scratch", 4),
    "cifar100-classification": ("C0S", "CIFAR-100 — Scratch", 5),
    "tiny-imagenet-classification": ("TIS", "Tiny ImageNet 200 — Scratch", 6),
    "oxford-pet-segmentation": ("OXS", "Oxford-IIIT Pet — Scratch", 7),
    "pascal-voc2012-segmentation": ("PVS", "Pascal VOC 2012 — Scratch", 8),
    "uavid-segmentation": ("UAS", "UAVid 2020 — Scratch", 9),
    "carpk-aerial-vehicle-pretrained": ("CRP", "CARPK — Pretrained", 10),
    "vedai-aerial-vehicle-pretrained": ("VDP", "VEDAI 512 — Pretrained", 11),
    "visdrone-aerial-vehicle-pretrained": ("VSP", "VisDrone2019-DET Vehicles — Pretrained", 12),
    "cifar10-classification-pretrained": ("C1P", "CIFAR-10 — Pretrained", 13),
    "cifar100-classification-pretrained": ("C0P", "CIFAR-100 — Pretrained", 14),
    "tiny-imagenet-classification-pretrained": ("TIP", "Tiny ImageNet 200 — Pretrained", 15),
    "oxford-pet-segmentation-resnet18-unet-pretrained": ("OXP", "Oxford-IIIT Pet — Pretrained", 16),
    "pascal-voc2012-segmentation-resnet18-unet-pretrained": ("PVP", "Pascal VOC 2012 — Pretrained", 17),
    "uavid-segmentation-resnet18-unet-pretrained": ("UAP", "UAVid 2020 — Pretrained", 18),
    "go-emotions-text-classification-scratch": ("GES", "GoEmotions — Scratch", 19),
    "go-emotions-text-classification-pretrained": ("GEP", "GoEmotions — Pretrained", 20),
    "multi-eurlex-text-classification-scratch": ("MES", "MultiEURLEX — Scratch", 21),
    "multi-eurlex-text-classification-pretrained": ("MEP", "MultiEURLEX — Pretrained", 22),
    "marc-multilingual-text-classification-scratch": ("MRS", "MARC Multilingual — Scratch", 23),
    "marc-multilingual-text-classification-pretrained": ("MRP", "MARC Multilingual — Pretrained", 24),
    "wikitext103-language-modeling-scratch": ("WTS", "WikiText-103 — Scratch", 25),
    "wikitext103-language-modeling-pretrained": ("WTP", "WikiText-103 — Pretrained", 26),
    "lm1b-shard-language-modeling-scratch": ("L1S", "One Billion Word Shard — Scratch", 27),
    "lm1b-shard-language-modeling-pretrained": ("L1P", "One Billion Word Shard — Pretrained", 28),
    "c4-en-shard-language-modeling-scratch": ("C4S", "C4 English Shard — Scratch", 29),
    "c4-en-shard-language-modeling-pretrained": ("C4P", "C4 English Shard — Pretrained", 30),
    "traffic-hourly-time-series-forecasting-scratch": ("TRS", "Traffic Hourly — Scratch", 31),
    "traffic-hourly-time-series-forecasting-pretrained": ("TRP", "Traffic Hourly — Pretrained", 32),
    "electricity-hourly-time-series-forecasting-scratch": ("ELS", "Electricity Hourly — Scratch", 33),
    "electricity-hourly-time-series-forecasting-pretrained": ("ELP", "Electricity Hourly — Pretrained", 34),
    "solar-10min-time-series-forecasting-scratch": ("SOS", "Solar 10-Minute — Scratch", 35),
    "solar-10min-time-series-forecasting-pretrained": ("SOP", "Solar 10-Minute — Pretrained", 36),
    "movielens25m-recommendation-scratch": ("MLS", "MovieLens 25M — Scratch", 37),
    "movielens25m-recommendation-pretrained": ("MLP", "MovieLens 25M — Pretrained", 38),
    "amazon-books-5core-recommendation-scratch": ("ABS", "Amazon Books 5-core — Scratch", 39),
    "amazon-books-5core-recommendation-pretrained": ("ABP", "Amazon Books 5-core — Pretrained", 40),
    "amazon-electronics-5core-recommendation-scratch": ("AES", "Amazon Electronics 5-core — Scratch", 41),
    "amazon-electronics-5core-recommendation-pretrained": ("AEP", "Amazon Electronics 5-core — Pretrained", 42),
    "minatar-breakout-reinforcement-learning-scratch": ("BRS", "MinAtar Breakout — Scratch", 43),
    "minatar-breakout-reinforcement-learning-pretrained": ("BRP", "MinAtar Breakout — Pretrained", 44),
    "minatar-seaquest-reinforcement-learning-scratch": ("SQS", "MinAtar Seaquest — Scratch", 45),
    "minatar-seaquest-reinforcement-learning-pretrained": ("SQP", "MinAtar Seaquest — Pretrained", 46),
    "minatar-asterix-reinforcement-learning-scratch": ("ATS", "MinAtar Asterix — Scratch", 47),
    "minatar-asterix-reinforcement-learning-pretrained": ("ATP", "MinAtar Asterix — Pretrained", 48),
}


def case_metadata(case_id: object) -> tuple[str, str, int]:
    identifier = str(case_id)
    return CASE_LABELS.get(identifier, (identifier[:3].upper(), identifier, 999))


def validate_case_labels() -> None:
    codes = [metadata[0] for metadata in CASE_LABELS.values()]
    if len(CASE_LABELS) != 48 or len(codes) != len(set(codes)):
        raise ValueError("The thesis case-label registry must contain 48 unique codes")
    if any(len(code) != 3 for code in codes):
        raise ValueError("Every thesis case code must contain exactly three characters")


validate_case_labels()
