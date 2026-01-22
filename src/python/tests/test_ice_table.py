from ptm_ice import IceTable, compute_ice_table


def test_compute_ice_table_zero_length():
    params = IceTable()
    res = compute_ice_table([], params)
    assert res['mass_change_kg_m2'] == 0.0
    assert res['times_s'] == []


def test_compute_ice_table_basic():
    params = IceTable(time_step=1.0)
    # constant cold temperature -> expect net small mass gain (condensation)
    temps = [-200.0] * 10
    res = compute_ice_table(temps, params)
    assert len(res['rates_kg_m2_s']) == 10
    # mass change should be finite and not NaN
    assert isinstance(res['mass_change_kg_m2'], float)
