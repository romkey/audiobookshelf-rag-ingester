from audiobookshelf_rag_ingester.linker import Linker


def test_links_unambiguous_aliases() -> None:
    linker = Linker(
        [
            ("asset", "laser-1", "the big laser"),
            ("location", "room-2", "Electronics Lab"),
            ("asset", "x", "CNC"),  # too short to match
            ("unknown", "y", "Something"),
        ]
    )
    links = linker.link("Use the Big Laser in the electronics lab, not the CNC.")
    assert links.ids == {"asset_ids": ["laser-1"], "location_ids": ["room-2"]}
    assert links.get("event_ids") == []


def test_ambiguous_aliases_need_confirmation() -> None:
    rows = [("asset", "a1", "Prusa"), ("asset", "a2", "Prusa")]
    assert Linker(rows).link("the Prusa").ids == {}
    confirmed = Linker(rows, lambda text, alias, kind, entity: entity == "a2")
    assert confirmed.link("the Prusa").ids == {"asset_ids": ["a2"]}


def test_no_aliases() -> None:
    assert Linker([]).link("anything").ids == {}
