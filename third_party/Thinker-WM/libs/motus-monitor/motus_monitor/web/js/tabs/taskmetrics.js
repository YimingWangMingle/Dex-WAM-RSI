import { api } from '../api.js';
import { store } from '../store.js';
const $=id=>document.getElementById('taskmetrics-'+id);
const training={total_loss:'Total loss',video_loss:'RGB loss',depth_loss:'Depth loss',action_loss:'Action loss'};
const validation={video_mse:'RGB MSE [0,1]',video_psnr:'RGB PSNR (dB)',depth_gray_mse:'Depth 编码 MSE',depth_mse_m2:'Depth MSE (m²)',depth_absrel:'Depth AbsRel',action_mse:'Action MSE（原生维度）'};
let chart,data,busy=false,version=0,last=0;
function options(node,items,empty){const old=node.value;node.replaceChildren();if(empty)node.add(new Option(empty,''));items.forEach(([k,v])=>node.add(new Option(v,k)));if([...node.options].some(o=>o.value===old))node.value=old;}
function mode(){return $('phase').value;}
function metrics(){return mode()==='train'?training:validation;}
function setup(){options($('metric'),Object.entries(metrics()));render();}
function render(){
 if(!data)return;
 options($('task'),data.tasks.map(t=>[t.task,t.task]),'全部任务（整体均值）');
 const phase=mode(),key=$('metric').value,curve=phase==='train'?data.train_curve:data.eval_curve;
 $('status').textContent=phase==='train'?
   (data.train_first_step==null?'尚无分任务训练 loss；历史全局 loss 不能归属到单个任务。':`训练记录 ${data.train_first_step}–${data.train_last_step}；下表统计最近 ${data.train_recent_steps} step`):
   (data.eval_last_step==null?'尚无 holdout 验证记录':`验证集：最新 step ${data.eval_last_step}；各任务按样本等权平均`);
 $('note').textContent=data.note;
 chart.resize();chart.setOption({title:{text:($('task').value||'全部任务整体均值')+' · '+metrics()[key],textStyle:{fontSize:15}},
  tooltip:{trigger:'axis'},grid:{left:85,right:55,top:65,bottom:40},xAxis:{type:'value',name:'step'},
  yAxis:{type:'value',scale:true},
  series:[{type:'line',connectNulls:false,showSymbol:phase!=='train',data:curve.map(p=>[p.step,p.metrics[key]?.value??null])}]},true);
 const keys=Object.keys(metrics());$('head').replaceChildren();
 ['任务',...Object.values(metrics()),phase==='train'?'Action 有效 batch / 样本数':'验证样本数'].forEach(v=>{const th=document.createElement('th');th.textContent=v;$('head').append(th);});
 $('rows').replaceChildren();const q=$('search').value.toLowerCase();
 const rows=data.tasks.filter(t=>t.task.toLowerCase().includes(q));
 rows.sort((a,b)=>(b[phase==='train'?'train':'validation'][key]?.mean??-Infinity)-(a[phase==='train'?'train':'validation'][key]?.mean??-Infinity));
 rows.forEach(t=>{const tr=document.createElement('tr');tr.title='点击查看该任务的历史曲线';tr.onclick=()=>{$('task').value=t.task;version++;load(true);};
  const name=document.createElement('td');name.textContent=t.task;tr.append(name);
  keys.forEach(k=>{const m=t[phase==='train'?'train':'validation'][k];const td=document.createElement('td');
   td.textContent=m?Number(m.mean).toPrecision(5):'—';td.title=m?(phase==='train'?`${m.batches} batch / ${m.samples} 样本`:`${m.n} 有效样本`):'无记录，不代表 0';tr.append(td);});
  const n=document.createElement('td');n.textContent=phase==='train'?(t.train.action_loss?`${t.train.action_loss.batches} / ${t.train.action_loss.samples}`:'—'):t.validation_samples;tr.append(n);$('rows').append(tr);
 });
}
async function load(force=false){if(busy||(!force&&Date.now()-last<30000))return;const run=$('run').value;if(!run)return;
 busy=true;const ticket=version;$('status').textContent='正在汇总任务指标…';
 try{const result=await api.get('/api/runs/'+encodeURIComponent(run)+'/task_metrics?task='+encodeURIComponent($('task').value));
  if(ticket!==version)return;data=result;last=Date.now();render();
 }catch(e){$('status').textContent='加载失败：'+e.message;}finally{busy=false;if(ticket!==version)load(true);}
}
export default {id:'taskmetrics',init(){
 chart=window.echarts.init($('chart'));
 $('run').onchange=()=>{version++;options($('task'),[],'全部任务（整体均值）');load(true);};
 $('task').onchange=()=>{version++;load(true);};$('phase').onchange=setup;$('metric').onchange=render;
 $('search').oninput=render;$('refresh').onclick=()=>load(true);setup();
 window.addEventListener('resize',()=>chart.resize());
},enter(){const old=$('run').value;options($('run'),store.state.runs.map(r=>[r.run_id,r.run_name||r.run_id]));
 if(!old&&store.state.runId)$('run').value=store.state.runId;chart.resize();load(true);
},leave(){},refresh(){},poll(){return load();}};
