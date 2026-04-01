from .posterior import *


def test_DET():
    det = DET(jnp.array(2.), r0=jnp.array(1.2))

    x = jnp.linspace(det.r0, 5, 10_000)
    assert jnp.isclose(jnp.trapezoid(det.pdf(x), x), 1.)

    assert jnp.array_equal(det.icdf(jnp.array([0., 1.])), jnp.array([det.r0, jnp.inf]))

def test_DET2():
    det = DET2(jnp.array(2.), jnp.array(.2), r0=jnp.array(1.2))

    x = jnp.linspace(det.r0, 30, 10_000)
    assert jnp.isclose(jnp.trapezoid(det.pdf(x), x), 1., atol=1e-6)

    assert jnp.isclose(det.icdf(jnp.array([0., .9999])), jnp.array([det.r0, 7.181044161412761])).all()

def test_ExpTruncatedPowerLaw():
    model = ExpTruncatedPowerLaw(jnp.array(2.1), jnp.array(.1), x0=jnp.array(.9))
    x = jnp.linspace(model.x0, 30, 10_000)
    assert jnp.isclose(jnp.trapezoid(model.pdf(x), x), 1.)

