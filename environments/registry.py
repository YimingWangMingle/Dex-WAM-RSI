"""Explicit platform factories; imports stay lazy so one simulator is optional."""
import importlib

_FACTORIES = {
    'robodojo': 'environments.robodojo.environment:RoboDojoEnvironment',
}

def register(name, factory):
    if not name or name in _FACTORIES or not callable(factory):
        raise ValueError('new unique platform name and callable factory required')
    _FACTORIES[name] = factory

def create(name, **kwargs):
    factory = _FACTORIES[name]
    if isinstance(factory,str):
        module, attr = factory.split(':')
        factory = getattr(importlib.import_module(module),attr)
    return factory(**kwargs)

def available():
    return tuple(sorted(_FACTORIES))
