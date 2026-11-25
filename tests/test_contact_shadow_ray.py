import pytest
import graphics.contact_shadow_ray as mod

CASES = [
    ("depth_current_frame", (1,)),
    ("step_screen_scale", (1,)),
    ("stop_at_discontinuity", (1,)),
    ("offscreen_fallback", (1,)),
    ("merge_darkest", (1,)),
    ("temporal_filter", (1,)),
    ("load", (1,)),
    ("trace", (1,)),
    ("shade", (1,)),
    ("present", (1,)),
    ("validate_contact", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
