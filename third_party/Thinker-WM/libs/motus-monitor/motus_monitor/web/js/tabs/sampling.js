import { api } from '../api.js';
import { store } from '../store.js';

let plots=[], data=null, busy=false, refreshed=0, generation=0;
const el=id=>document.getElementById('sampling-'+id);
function options(node, rows, empty) {
  const current=node.value;
  node.replaceChildren();
  if(empty) node.add(new Option(empty,''));
  rows.forEach(([value,label])=>node.add(new Option(label,value)));
  if([...node.options].some(x=>x.value===current)) node.value=current;
}
function runs() {
  const previous=el('run').value;
  options(el('run'),store.state.runs.map(r=>[r.run_id,r.run_name||r.run_id]));
  if(!previous && store.state.runId) el('run').value=store.state.runId;
}
function chart() {
  if(!plots.length) plots=['distribution'].map(id=>window.echarts.init(el(id)));
  plots.forEach(p=>p.resize());
}
function render() {
  chart();
  const tasks=data.tasks;
  el('distribution').style.height=Math.max(300,tasks.length*30+100)+'px';
  plots[0].resize();
  options(el('task'),tasks.map(t=>[t.task,t.task]),'全部任务');
  options(el('episode'),el('task').value?data.episodes.map(e=>[e.episode_id,e.episode_id+' · '+e.task]):[],
    el('task').value?'全部 episode':'先选任务以筛选 episode');
  const last=data.curve.at(-1);
  el('summary').textContent=last ?
    `记录 step ${data.first_step}–${data.last_step} · ${data.ranks.length} 个 rank · 累计 ${last.samples.toLocaleString()} 次 · 去重 ${last.unique_windows.toLocaleString()} 窗口 · 覆盖率 ${last.coverage==null?'未知':(last.coverage*100).toFixed(2)+'%'}` : '所选范围暂无已记录窗口';
  el('note').textContent=data.note+(data.metadata_available?'':' 尚无任务索引，任务名/覆盖率待补充。');
  plots[0].setOption({title:{text:'各任务累计采样次数',textStyle:{fontSize:15}},
    tooltip:{trigger:'axis'},grid:{left:235,right:75,top:55,bottom:45},
    xAxis:{type:'value',name:'窗口次数'},
    yAxis:{type:'category',inverse:true,data:tasks.map(t=>t.task),axisLabel:{fontSize:13,width:210,overflow:'truncate'}},
    series:[{name:'已记录窗口次数',type:'bar',data:tasks.map(t=>t.samples),itemStyle:{color:'#5276d9'}}]},true);
  table();
}
function table() {
  if(!data)return;
  const query=el('search').value.toLowerCase();
  let rows=data.episodes.filter(e=>(e.episode_id+' '+e.task).toLowerCase().includes(query));
  const sort=el('sort').value;
  rows.sort((a,b)=>sort==='coverage'?(a.coverage??-1)-(b.coverage??-1):
    sort==='most'?b.samples-a.samples:a.samples-b.samples);
  el('rows').replaceChildren();
  rows.slice(0,200).forEach(e=>{
    const tr=document.createElement('tr');
    [e.task,e.episode_id,e.samples,e.unique_windows,e.total_windows??'未知',
      e.coverage==null?'未知':(e.coverage*100).toFixed(2)+'%',e.last_step??'未见'].forEach(v=>{
      const td=document.createElement('td');td.textContent=v;td.style.padding='12px 14px';tr.append(td);
    });
    tr.style.cursor='pointer';tr.title='点击筛选此 episode 的采样统计';
    tr.onclick=()=>{el('task').value=e.task;options(el('episode'),[[e.episode_id,e.episode_id]],'全部 episode');
      el('episode').value=e.episode_id;generation++;load(true);};el('rows').append(tr);
  });
  el('table-note').textContent=`匹配 ${rows.length} 条，显示前 ${Math.min(200,rows.length)} 条；可搜索 episode，点击行筛选统计。`;
}
async function load(force=false) {
  if(busy || (!force&&Date.now()-refreshed<30000))return;
  const run=el('run').value;if(!run)return;
  busy=true;const ticket=generation;
  el('status').textContent='正在统计；首次读取历史记录可能需要一些时间…';
  try {
    const qs=new URLSearchParams({task:el('task').value,episode:el('episode').value});
    const result=await api.get('/api/runs/'+encodeURIComponent(run)+'/sampling?'+qs);
    if(ticket!==generation)return;
    data=result;render();refreshed=Date.now();el('status').textContent='已更新 '+new Date().toLocaleTimeString();
  } catch(e) {el('status').textContent='加载失败：'+e.message;}
  finally {busy=false;if(ticket!==generation)load(true);}
}
function changed(clearTask=false) {
  generation++;if(clearTask)options(el('task'),[],'全部任务');
  options(el('episode'),[],'全部 episode');data=null;load(true);
}
export default {id:'sampling',init(){
  el('run').onchange=()=>changed(true);el('task').onchange=()=>changed();
  el('episode').onchange=()=>{generation++;load(true);};el('refresh').onclick=()=>load(true);
  el('search').oninput=table;el('sort').onchange=table;
  window.addEventListener('resize',()=>plots.forEach(p=>p.resize()));
  store.on('run',()=>{if(store.state.tab==='sampling'){runs();el('run').value=store.state.runId;changed(true);}});
},enter(){runs();chart();load(true);},leave(){},refresh(){},poll(){return load();}};
