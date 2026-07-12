from sirrobin.physics.config import LocomotionConfig


def test_unit_chain_and_hash_are_frozen():
    config = LocomotionConfig()
    config.validate()
    assert config.kg_per_sim_mass == config.rho_water / config.rho_neutral_gene == 250.0
    assert config.s_slot == 17
    assert len(config.sha256()) == 64
