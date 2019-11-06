from phi.physics.field import Field, math, GeometryMask
from .physics import State, struct, staticshape


GROW = 'grow'
ADD = 'add'
FIX = 'fix'


class FieldEffect(State):

    def __init__(self, field, targets, mode=GROW, tags=('effect',), **kwargs):
        tags = tuple(tags) + tuple('%s_effect' % target for target in targets)
        State.__init__(**struct.kwargs(locals()))

    @struct.attr()
    def field(self, field):
        assert isinstance(field, Field)
        return field

    @struct.prop()
    def mode(self, mode):
        assert mode in (GROW, ADD, FIX)
        return mode

    @struct.prop()
    def targets(self, targets):
        return tuple(targets)

    def __repr__(self):
        return '%s(%s to %s)' % (self.mode, self. field, self.targets)


def effect_applied(effect, field, dt):
    resampled = effect.field.at(field)
    if effect._mode == GROW:
        dt = math.cast(dt, resampled.dtype)
        return field + resampled * dt
    elif effect._mode == ADD:
        return field + resampled
    elif effect._mode == FIX:
        assert effect.field.bounds is not None
        mask = effect.field.bounds.value_at(field.points.data)
        return field * (1 - mask) + resampled * mask
    else:
        raise ValueError('Invalid mode: %s' % effect.mode)


Inflow = lambda geometry, rate=1.0:\
    FieldEffect(GeometryMask('inflow', [geometry], rate), ('density',), GROW, tags=('inflow', 'effect'))
Fan = lambda geometry, acceleration:\
    FieldEffect(GeometryMask('fan', [geometry], acceleration), ('velocity',), GROW, tags=('fan', 'effect'))
ConstantDensity = lambda geometry, density:\
    FieldEffect(GeometryMask('constant-density', [geometry], density), ('density',), FIX)
ConstantTemperature = lambda geometry, temperature:\
    FieldEffect(GeometryMask('constant-temperature', [geometry], temperature), ('temperature',), FIX)
HeatSource = lambda geometry, rate:\
    FieldEffect(GeometryMask('heat-source', [geometry], rate), ('temperature',), GROW)
ColdSource = lambda geometry, rate:\
    FieldEffect(GeometryMask('heat-source', [geometry], -rate), ('temperature',), GROW)


class Gravity(State):

    def __init__(self, gravity=-9.81, **kwargs):
        tags = ['gravity']
        State.__init__(**struct.kwargs(locals()))

    @struct.prop()
    def gravity(self, gravity):
        assert gravity is not None
        return gravity

    def __add__(self, other):
        if other is 0:
            return self
        assert isinstance(other, Gravity)
        if self._batch_size is not None:
            assert self._batch_size == other._batch_size
        # Add gravity
        if math.is_scalar(self.gravity) and math.is_scalar(other.gravity):
            return Gravity(self.gravity + other.gravity)
        else:
            rank = staticshape(other.gravity)[-1] if math.is_scalar(self.gravity) else staticshape(self.gravity)[-1]
            sum_tensor = gravity_tensor(self, rank) + gravity_tensor(other, rank)
            return Gravity(sum_tensor)

    __radd__ = __add__


def gravity_tensor(gravity, rank):
    if isinstance(gravity, Gravity):
        gravity = gravity.gravity
    if math.is_scalar(gravity):
        return math.expand_dims([gravity] + [0] * (rank-1), 0, rank+1)
    else:
        assert staticshape(gravity)[-1] == rank
        return math.expand_dims(gravity, 0, rank+2-len(staticshape(gravity)))