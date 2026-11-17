import pytest

from audio.mixer import (
    BusGraph,
    Mixer,
    db_to_linear,
    distance_gain,
    linear_to_db,
    ms_to_samples,
)


def make_mixer():
    mixer = Mixer(sample_rate=48000, block_size=256)
    mixer.add_bus("music")
    mixer.add_bus("voice")
    mixer.add_bus("ui")
    return mixer


def test_db_to_linear_minus_six():
    assert db_to_linear(-6) == pytest.approx(0.501187, rel=1e-3)


def test_db_to_linear_plus_three():
    assert db_to_linear(3) == pytest.approx(1.412538, rel=1e-3)


def test_linear_to_db_half():
    assert linear_to_db(0.5) == pytest.approx(-6.0206, rel=1e-3)


def test_linear_to_db_two():
    assert linear_to_db(2.0) == pytest.approx(6.0206, rel=1e-3)


def test_ms_to_samples_rounds_half_up():
    assert ms_to_samples(2.6, 48000) == 125


def test_distance_gain_uses_inverse_distance():
    assert distance_gain(10.0) == pytest.approx(0.1)


def test_bus_graph_orders_sources_before_destinations():
    graph = BusGraph()
    graph.add_bus("reverb")
    graph.add_bus("music")
    graph.add_send("music", "reverb")
    assert graph.order() == ["music", "reverb"]


def test_bus_graph_detects_cycle():
    graph = BusGraph()
    graph.add_bus("a")
    graph.add_bus("b")
    graph.add_send("a", "b")
    graph.add_send("b", "a")
    assert graph.has_cycle() is True


def test_send_applies_gain():
    mixer = make_mixer()
    mixer.add_send("music", "voice", gain_db=-6)
    output = mixer.process({"music": [1.0] * 256})
    assert output["voice"][0] == pytest.approx(0.501187, rel=1e-3)


def test_send_applies_delay():
    mixer = Mixer(sample_rate=48000, block_size=16)
    mixer.add_bus("music")
    mixer.add_bus("voice")
    mixer.add_send("music", "voice", delay_ms=0.1)
    output = mixer.process({"music": [1.0] * 16})
    assert output["voice"][5] == pytest.approx(1.0)
    assert output["voice"][4] == pytest.approx(0.0)
    assert output["voice"][0] == pytest.approx(0.0)


def test_sidechain_ducks_all_targets():
    mixer = make_mixer()
    mixer.add_sidechain("music", ["voice", "ui"], threshold_db=-20.0)
    output = mixer.process(
        {
            "music": [1.0] * 256,
            "voice": [1.0] * 256,
            "ui": [1.0] * 256,
        }
    )
    assert output["voice"][0] == pytest.approx(0.5)
    assert output["ui"][0] == pytest.approx(0.5)


def test_sidechain_does_not_duck_quiet_source():
    mixer = make_mixer()
    mixer.add_sidechain("music", ["voice"], threshold_db=-20.0)
    output = mixer.process({"music": [0.01] * 256, "voice": [1.0] * 256})
    assert output["voice"][0] == pytest.approx(1.0)


def test_voice_selection_respects_priority():
    mixer = make_mixer()
    voices = [
        {"name": "low", "priority": 1, "distance": 1.0},
        {"name": "high", "priority": 3, "distance": 5.0},
        {"name": "mid", "priority": 2, "distance": 2.0},
    ]
    assert [voice["name"] for voice in mixer.select_voices(voices, 2)] == ["high", "mid"]


def test_voice_selection_prefers_closer_voice_at_same_priority():
    mixer = make_mixer()
    voices = [
        {"name": "far", "priority": 1, "distance": 3.0},
        {"name": "close", "priority": 1, "distance": 1.0},
        {"name": "mid", "priority": 1, "distance": 2.0},
    ]
    assert [voice["name"] for voice in mixer.select_voices(voices, 2)] == ["close", "mid"]


def test_send_to_unknown_bus_is_rejected():
    mixer = make_mixer()
    with pytest.raises(ValueError):
        mixer.add_send("music", "missing")
