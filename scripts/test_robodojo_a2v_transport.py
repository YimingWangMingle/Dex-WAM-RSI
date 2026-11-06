"""Real localhost XPL WebSocket and codecs, CPU model double, no live service."""
import asyncio
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_framework_native import Strict, Env
from environments.robodojo.framework import run_framework_episode
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from client_server.ws.model_client import WsModelClient
from XPolicyLab.utils import process_data


async def main():
    with tempfile.TemporaryDirectory() as tmp:
        policy=Strict(tmp)
        server=PolicyServer(policy,PolicyServerConfig(host='127.0.0.1',port=0))
        await server.start()
        def client_test():
            client=WsModelClient(url=server.url,evaluation_id='a2v-interface',trial_id='cpu',
                                 request_timeout_s=30)
            try:
                env=Env(stop=3)
                result=run_framework_episode(env,root=Path(tmp)/'run',client=client,max_steps=8,
                    reviewer=lambda packet:dict(candidate_id=2,execute_steps=8))
                assert result['executed_steps']==3
                assert policy.measured==[0,1,2,3]
                assert policy.next_decision==3 and policy.last_command is None
                assert all(abs(row[0]-.114)<1e-6 for row in env.actions)
                try:
                    client.call('framework_select',dict(review_id='stale',candidate_id=2,execute_steps=8))
                except Exception:
                    pass
                else:
                    raise AssertionError('duplicate selection accepted')
                print('PASS unified runner over real WebSocket: identity, replay, 3 candidates, selection, partial feedback, stale rejection')
            finally:
                client.close()
        try:
            with patch.object(process_data,'unpack_robot_state',lambda a,*args,**kw:a.copy()):
                await asyncio.to_thread(client_test)
        finally:
            await server.stop()


if __name__=='__main__':asyncio.run(main())
