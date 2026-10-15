from .structure import build_structure_domain
from .thermal import build_thermal_domain

def build_domain(name, cfg):
    if name == "structure":
        return build_structure_domain(cfg)
    if name == "thermal":
        return build_thermal_domain(cfg)
    if name == "power":
        from .power import build_power_domain
        return build_power_domain(cfg)
    if name == "water":
        from .water import build_water_domain
        return build_water_domain(cfg)
    raise KeyError(name)
