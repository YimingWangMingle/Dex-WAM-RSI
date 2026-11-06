"""Provider interface contracts via real localhost HTTP, no external API calls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from agents.llm_clients import DeepSeekClient, KimiClient, ModelAPIError, client_from_env


class Tests(unittest.TestCase):
    def setUp(self):
        self.received=[];self.status=200;self.redirect=None
        self.answer=dict(id='fixture',choices=[dict(finish_reason='tool_calls',message=dict(
            role='assistant',content=None,reasoning_content='opaque-provider-field',
            tool_calls=[dict(id='call-1',type='function',function=dict(name='stop',arguments='{}'))]))])
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.received.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(owner.status)
                self.send_header('Content-Type','application/json')
                if owner.redirect:self.send_header('Location',owner.redirect)
                self.end_headers();self.wfile.write(json.dumps(owner.answer).encode())
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)+'/v1'
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.messages=[dict(role='user',content=[dict(type='text',text='ICL / HMR historical context'),
            dict(type='image_url',image_url=dict(url='data:image/png;base64,fixture'))])]
        self.tools=[dict(type='function',function=dict(name='stop',parameters=dict(type='object',properties={})))]
    def client(self,cls=DeepSeekClient,**kw):return cls('fixture-secret',base_url=self.base,**kw)
    def test_deepseek_non_thinking_payload_and_vision(self):
        client=self.client();before=copy.deepcopy(self.messages)
        client.complete(self.messages,self.tools);path,payload=self.received[-1]
        self.assertEqual(path,'/v1/chat/completions');self.assertEqual(payload['tool_choice'],'required')
        self.assertEqual(payload['thinking'],dict(type='disabled'))
        self.assertEqual(payload['messages'],self.messages);self.assertEqual(self.messages,before)
        self.assertNotIn('reasoning_effort',payload);self.assertEqual(client.last_metadata['provider'],'deepseek')
    def test_deepseek_thinking_uses_auto(self):
        self.client(thinking=True,reasoning_effort='high').complete(self.messages,self.tools)
        p=self.received[-1][1];self.assertEqual(p['tool_choice'],'auto');self.assertEqual(p['reasoning_effort'],'high')
    def test_kimi_k3_parameter_contract(self):
        self.client(KimiClient,max_tokens=2048).complete(self.messages,self.tools)
        p=self.received[-1][1]
        self.assertEqual(p['model'],'kimi-k3');self.assertEqual(p['max_completion_tokens'],2048)
        for field in ('max_tokens','thinking','temperature','top_p'):self.assertNotIn(field,p)
        self.assertEqual(p['messages'],self.messages)
    def test_restores_reasoning_for_eef_reconstructed_tool_message(self):
        c=self.client(thinking=True);r=c.complete(self.messages,self.tools)
        assistant=copy.deepcopy(r['choices'][0]['message']);assistant.pop('reasoning_content')
        history=[*self.messages,assistant,dict(role='tool',tool_call_id='call-1',content='executed=0')]
        c.complete(history,self.tools)
        self.assertEqual(self.received[-1][1]['messages'][1]['reasoning_content'],'opaque-provider-field')
        self.assertNotIn('reasoning_content',assistant)
    def test_different_arguments_do_not_recover_other_response(self):
        c=self.client(KimiClient);r=c.complete(self.messages,self.tools)
        m=copy.deepcopy(r['choices'][0]['message']);m.pop('reasoning_content');m['tool_calls'][0]['function']['arguments']='{"changed":true}'
        c.complete([*self.messages,m],self.tools)
        self.assertNotIn('reasoning_content',self.received[-1][1]['messages'][1])
    def test_http_error_not_retried_or_echoed(self):
        self.status=429;self.answer={'error':{'message':'fixture-secret private prompt'}}
        with self.assertRaises(ModelAPIError) as error:self.client().complete(self.messages,self.tools)
        self.assertNotIn('fixture-secret',str(error.exception));self.assertNotIn('private',str(error.exception));self.assertEqual(len(self.received),1)
    def test_redirect_is_not_followed(self):
        self.status=307;self.redirect=self.base+'/other'
        with self.assertRaises(ModelAPIError):self.client().complete(self.messages,self.tools)
        self.assertEqual(len(self.received),1)
    def test_truncated_response_rejected(self):
        self.answer['choices'][0]['finish_reason']='length'
        with self.assertRaises(ModelAPIError):self.client().complete(self.messages,self.tools)
    def test_env_is_explicit_and_does_not_fallback(self):
        c=client_from_env(dict(RSI_LLM_PROVIDER='kimi',MOONSHOT_API_KEY='fixture',RSI_LLM_MODEL='kimi-k3',RSI_LLM_BASE_URL=self.base))
        self.assertIsInstance(c,KimiClient)
        for e in ({},{'RSI_LLM_PROVIDER':'other'},{'RSI_LLM_PROVIDER':'kimi','DEEPSEEK_API_KEY':'fixture'}):
            with self.assertRaises(ValueError):client_from_env(e)
    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):self.client(KimiClient,model='kimi-k2.6')
        with self.assertRaises(ValueError):self.client(timeout=0)
        with self.assertRaises(ValueError):self.client(max_tokens=True)
        with self.assertRaises(ValueError):KimiClient('fixture',base_url='http://remote.example/v1')
    def test_bridge_clients_do_not_share_provider_conversation_state(self):
        from environments.robodojo.runtime.robodojo_codex_bridge import controller_clients, FileController
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            a,b=controller_clients(root,dict(RSI_LLM_PROVIDER='kimi',MOONSHOT_API_KEY='fixture'))
            self.assertIsInstance(a,KimiClient);self.assertIsNot(a,b)
            a,b=controller_clients(root,{})
            self.assertIsInstance(a,FileController);self.assertIs(a,b)

    def test_no_tools_and_legacy_import(self):
        from agents.deepseek import DeepSeekClient as Legacy, DeepSeekError
        self.assertIs(Legacy,DeepSeekClient);self.assertIs(DeepSeekError,ModelAPIError)
        self.client().complete(self.messages,[])
        self.assertNotIn('tool_choice',self.received[-1][1])

class OfficialControlsTests(unittest.TestCase):
    def test_task_native_limits_and_official_defaults(self):
        from environments.robodojo.runtime.robodojo_codex_bridge import official_eef_config
        for steps in (400, 800, 1900):
            cfg=official_eef_config({},steps)
            self.assertEqual(cfg['RSI_MAX_CONTROL_STEPS'],str(steps))
            self.assertEqual(cfg['L3_INSPECT_MAX_LLM_CALLS'],'170')
            self.assertEqual(cfg['L3_INSPECT_USE_RECIPE'],'1')
            self.assertEqual(cfg['L3_INSPECT_KEEP_ALL_IMAGES'],'0')
            self.assertEqual(cfg['L3_INSPECT_IMAGE_HORIZON'],'2')

    def test_inherited_ablation_settings_are_rejected(self):
        from environments.robodojo.runtime.robodojo_codex_bridge import official_eef_config
        for env in ({'L3_INSPECT_MAX_LLM_CALLS':'40'},
                    {'L3_INSPECT_USE_RECIPE':'0'},
                    {'RSI_MAX_CONTROL_STEPS':'800'},
                    {'RSI_CONTROL_MODE':'hybrid'},
                    {'RSI_EXPERIENCE_CATALOG':'fixture'},
                    {'L3_INSPECT_EEF_JITTER_DX':'0.1'},
                    {'L3_INSPECT_DISABLE_GIVE_UP':'1'}):
            with self.subTest(env=env),self.assertRaises(ValueError):
                official_eef_config(env,400)

    def test_file_baseline_keeps_policy_selected_images(self):
        from environments.robodojo.runtime.robodojo_codex_bridge import controller_clients
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            a,b=controller_clients(root,{'RSI_EVAL_PROTOCOL':'official_eef'})
            self.assertIs(a,b)
            self.assertTrue(a.retain_policy_images)

if __name__=='__main__':unittest.main()
