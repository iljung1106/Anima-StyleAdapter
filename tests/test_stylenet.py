import torch

from anima_style_data.stylenet import controlled_style_ranking, parse_stylenet_member


def test_parse_stylenet_member_contract():
    parsed = parse_stylenet_member(
        "images/Group_0003_amiya__arknights_/00_Original_example.artist.jpg"
    )

    assert parsed == {
        "group_index": 3,
        "subject": "amiya__arknights_",
        "candidate_rank": 0,
        "candidate_artist": "example.artist",
        "is_original": True,
        "extension": "jpg",
    }
    assert parse_stylenet_member("README.txt") is None


def test_controlled_ranking_uses_other_groups_as_references():
    rows = []
    values = []
    for group in range(3):
        for candidate in range(4):
            rows.append(
                {
                    "id": len(rows),
                    "feature_index": len(rows),
                    "group_key": f"artist:{group}",
                    "group_index": group,
                    "shard_artist": "artist",
                    "is_original": candidate == 0,
                    "global_exact_duplicate": False,
                }
            )
            values.append([1.0, 0.0] if candidate == 0 else [0.0, 1.0])

    metrics = controlled_style_ranking(torch.tensor(values), rows, reference_count=1)

    assert metrics["queries"] == 2
    assert metrics["top1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["margin"] == 1.0


def test_controlled_ranking_can_return_query_decisions():
    rows = []
    values = []
    for group in range(3):
        for candidate in range(4):
            rows.append(
                {
                    "id": len(rows),
                    "feature_index": len(rows),
                    "group_key": f"artist:{group}",
                    "group_index": group,
                    "shard_artist": "artist",
                    "is_original": candidate == 0,
                    "global_exact_duplicate": False,
                }
            )
            values.append([1.0, 0.0] if candidate == 0 else [0.0, 1.0])

    metrics, correct = controlled_style_ranking(
        torch.tensor(values), rows, reference_count=1, return_correct=True
    )

    assert metrics["queries"] == 2
    assert correct.tolist() == [True, True]
