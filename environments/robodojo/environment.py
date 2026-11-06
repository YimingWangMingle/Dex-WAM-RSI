"""RoboDojo lifecycle and measured observation adapter; no WAM imports."""
import numpy as np
from environments import Observation, Outcome

from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect.types import ActionSpace, JOINT_CHANNELS
CHANNELS = tuple(name for name, size in JOINT_CHANNELS)
SIZES = tuple(size for name, size in JOINT_CHANNELS)


class RoboDojoEnvironment:
    def __init__(self, native, *, contract, reset, close, revision):
        if contract.dimension != 14 or contract.representation != 'absolute':
            raise ValueError('this RoboDojo binding requires absolute joint14')
        self.native, self.contract = native, contract
        self._reset, self._close, self.revision = reset, close, revision
        self.step = 0

    def identity(self):
        return dict(platform='robodojo', revision=self.revision)

    def reset(self, case):
        self._reset(self.native, case)
        self.step = 0

    def observe(self):
        raw = self.native.get_obs()
        if float(raw['additional_info']['frequency']) != self.contract.control_hz:
            raise ValueError('RoboDojo observation frequency differs from action contract')
        state = []
        for key, n in zip(CHANNELS, SIZES):
            part = np.asarray(raw['state'][key]).reshape(-1)
            if len(part) != n:
                raise ValueError('wrong RoboDojo joint channel shape')
            state.extend(float(x) for x in part)
        images = {}
        for key, view in raw['vision'].items():
            if 'color' not in view:
                continue
            rgb = view['color']
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
                from XPolicyLab.utils.process_data import decode_image_bit
                rgb = np.asarray(decode_image_bit(rgb))
            images[key] = rgb
        return Observation(raw['instruction'], tuple(state), images, self.step,
                           self.step / self.contract.control_hz)

    def execute(self, row):
        values = dict(ActionSpace().decode(row).data)
        self.native.take_action(values)
        self.step += 1

    def is_terminal(self):
        return bool(self.native.is_episode_end())

    def outcome(self):
        success = (bool(self.native.success[0] and self.native.end_flag[0])
                   if hasattr(self.native,'success') and hasattr(self.native,'end_flag') else None)
        return Outcome(success, self.is_terminal())

    def close(self):
        self._close(self.native)
