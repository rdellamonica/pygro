import pytest
import pygro

base_matric_initializer = dict(
    name = "Test Schwarzschild",
    coordinates = ["t", "r", "theta", "phi"],
    transform_functions = [
        "t",
        "r*sin(theta)*cos(phi)",
        "r*sin(theta)*sin(phi)",
        "r*cos(theta)"
    ],
    line_element = "-(1-2*M/r)*dt**2+1/(1-2*M/r)*dr**2+r**2*(dtheta**2+sin(theta)**2*dphi**2)"
)

def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def test_line_element_base_inizialization():
    initializer = base_matric_initializer.copy()

    metric = pygro.Metric(
        M = 1,
        **initializer,
    )
    
@pytest.mark.parametrize("initializer", [
    without(base_matric_initializer, 'name'),
    without(base_matric_initializer, 'coordinates'),
])
def test_line_element_missing_arg(initializer):
    
    with pytest.raises(TypeError):
        metric = pygro.Metric(
            M = 1,
            **initializer
        )

def test_line_element_missing_coordinate():
    
    initializer = base_matric_initializer.copy()
    initializer.update(coordinates = ["t", "r", "theta"])
    
    with pytest.raises(ValueError):
        metric = pygro.Metric(
            M = 1,
            **initializer
        )
