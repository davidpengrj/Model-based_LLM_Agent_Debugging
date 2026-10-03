'use strict';

const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (value, digits=2) => Number(value).toFixed(digits);
const state = {lang:'en', stage:0, event:2, data:null, case:null, evidenceMode:'final', inspectedState:null, inspectManual:false};
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const copy = {
  scrubLabel:['Explore the six events','拖动查看六个事件'], playbackScope:['Replay the saved trace. The final prediction uses all six events.','回放已保存的轨迹；最终预测基于完整的六个事件。'], stateExplorer:['Explore the behavioral model','探索行为模型'], stateHint:['Select a state to inspect its saved signals','点击状态，查看它的已保存信号'], inspectState:['Inspect state','查看状态'], inspectedState:['Selected state','选中的状态'], stateMembership:['Membership at this event','当前事件的软状态质量'], initialFrequency:['Initial frequency','初始状态频率'], transitionFrequency:['Training transition frequency','训练中的状态转移频率'], inspectorNote:['These are saved state memberships and transition frequencies. Scoring combines all soft states; state IDs identify learned clusters.','这里展示已保存的软状态质量与转移频率。评分结合全部软状态，状态编号标识学习到的聚类。'],
  contentView:['Content evidence','内容证据'], finalView:['Final localization','最终定位'], contentViewHeading:['Content evidence across the trace','轨迹中的内容证据'], contentViewNote:['Content evidence peaks at event 04. Explore the final localization to see how incoming surprise changes the readout.','内容证据在事件 04 达到峰值。切换到最终定位，查看进入事件的转移惊讶度如何影响读出。'], finalViewNote:['The same method combines content and incoming surprise within the attributed agent. The final step mass peaks at event 03.','同一方法在已归因的智能体内，结合内容证据与传入转移惊讶度。最终步骤质量在事件 03 达到峰值。'],
  editionLabel:['Research project','研究项目'], editionType:['Interactive case study','交互式研究展示'], finalCore:['The final method','最终方法'], latentStates:['latent states','个隐状态'], softTransitions:['Soft-state transitions','软状态转移'], sourceCode:['Research repository ↗','研究代码仓库 ↗'],
  stage1Caption:['Agent / Action / State','智能体／动作／状态'], stage2Caption:['TF-IDF → PCA → GMM','TF-IDF → PCA → GMM'], stage3Caption:['Content + transition','内容证据＋转移信号'], stage4Caption:['WHO → WHEN','WHO → WHEN'],
  brandSub:['An interactive research project','研究项目 · 交互式方法展示'], navMethod:['Method','方法流程'], navCase:['Walkthrough','真实案例'], navEvidence:['Evidence','评测证据'],
  heroKicker:['Research in multi-agent systems','多智能体系统 · 研究项目'], heroTitle:['Model-based<br><span>Agent Debugging.</span>','多智能体系统的<br><span>模型化故障定位。</span>'], heroDescription:['Identify the agent behind a failure and locate its position in the trace, through semantic evidence and a model of behavior.','把运行日志转成可解释的语义与行为证据，定位应归因的智能体，以及轨迹中出错的位置。'], exploreMethod:['Explore the method ↘','了解方法 ↘'], followCase:['Follow a real trace →','跟着真实案例走一遍 →'], scopeWho:['Responsible agent','责任智能体'], scopeWhen:['Error position','错误位置'], behaviorMap:['A behavioral model, made visible','把行为模型变成可见的证据'], recordedRun:['Recorded run','真实运行记录'], graphNote:['State IDs are learned clusters, not error labels.','状态编号代表学习到的聚类，不是错误标签。'], agentOutput:['Agent attribution','智能体归因'], tvAgent:['TV series expert','电视剧领域专家'], positionOutput:['Annotated & predicted event','标注与预测一致的事件'], eventWord:['Event','事件'], matched:['Matched','命中'],
  problemLabel:['The research question','研究问题'], problemTitle:['A final answer hides the path that produced it.','最终答案，无法呈现产生它的整条路径。'], problemBody:['An agent makes a claim. Another builds on it. A verifier repeats it. Debugging requires tracing the failure through these events: which agent should be attributed, and which event should be localized?','一个智能体提出判断，后续步骤沿用这个判断，验证者再重复结论。定位失败需要回到这条事件链：应归因于哪个智能体，又该定位到哪一步？'], sequence1:['A claim enters the trace','判断进入轨迹'], sequence2:['Later steps build on it','后续步骤继续推演'], sequence3:['The answer inherits it','最终答案继承判断'],
  methodLabel:['The final method','最终方法'], methodTitle:['From a trace to an inspectable decision.','从运行轨迹，到可核查的定位决策。'], methodIntro:["Follow the four stages. Each view uses the same recorded event and the final method's saved signals.",'点击四个阶段，查看同一事件如何被表示、建模、评分和定位。所有数值来自最终方法的保存记录。'], stage1:['Structure the trace','结构化轨迹'], stage2:['Model behavior','建模行为'], stage3:['Score the evidence','计算证据'], stage4:['Attribute & locate','归因与定位'], fitNote:['Training estimates the frequency channels and state model. Inference reads the visible trace; the evaluation annotation is separate.','训练阶段估计词频通道与状态模型；推理阶段读取可见轨迹，评测标注单独保存。'], inspectCase:['Inspect the recorded case ↓','进入真实案例 ↓'],
  caseLabel:['A real trace','真实案例'], caseTitle:['Follow the decision, event by event.','沿着每一步，理解模型的决策。'], caseIntro:['Six events, three agents, one dataset-annotated first error. Click an event to inspect its trace, representation and evidence.','六个事件、三个智能体、一个由数据集标注的首次错误。点击任意事件，查看原始内容、语义表示和定位依据。'], caseIndexNote:['Events 01–06 correspond to source steps 0–5.','界面事件 01–06 对应源数据步骤 0–5。'], taskLabel:['Task','任务'], caseTask:['Find the first name of the Magda M. character played by the actor who portrayed Ray in the Polish adaptation of Everybody Loves Raymond.','找出在波兰版《人人都爱雷蒙德》中饰演 Ray 的演员，在《Magda M.》中扮演角色的名字。'], visibleEvent:['Visible event','可见事件'], originalLog:['Read the original log','查看原始日志'], semanticTriple:['Canonical representation','规范三元组表示'], eventEvidence:['Evidence at this event','当前事件的证据'], savedSignals:['Saved model signals','已保存的模型信号'], stepDistribution:['Step evidence after attribution','智能体归因后的步骤证据'], normalizedEvidence:['Normalized mass','归一化质量'], massNote:['Normalized evidence is a model readout, not a calibrated probability of real-world failure.','归一化证据是模型读出，不代表经过校准的现实错误概率。'], caseOutcome:['The final method matches the annotated agent and event.','最终方法与数据集标注的智能体、事件一致。'], caseTakeaway:['The third event receives less content probability than the fourth, but much higher incoming transition surprise. The behavioral signal shifts localization back to the annotated first error.','第 3 个事件的内容概率略低于第 4 个事件，但进入它的转移惊讶度明显更高。行为信号把定位拉回了数据集标注的首次错误。'], caseQualification:['This is one successful recorded example of the same core method on Who&When. It illustrates the decision process; the aggregate evaluation below uses Who&When Pro.','这是同一核心方法在 Who&When 上的一条成功记录，用于展示决策过程。下方总体评测使用 Who&When Pro。'],
  choicesLabel:['Key design choices','设计选择'], choicesTitle:['The reasoning behind the architecture.','为什么这样设计方法。'], choice1Title:['Preserve agent, action and state.','保留智能体、动作与状态。'], choice1Body:['Canonical triples turn verbose logs into a consistent representation while retaining task-affecting details. The extraction sees visible content, without the error annotation.','规范三元组将冗长日志转成一致的表示，保留影响任务的细节。语义提取只读取可见内容，不读取错误标注。'], choice2Title:['Read content and behavior together.','同时读取内容与行为证据。'], choice2Body:['Error/Normal token frequencies and Action–State relations supply content evidence. The DTMC supplies incoming transition surprise from a fold-local behavior model.','错误／正常词频与动作—状态关系提供内容证据，训练划分内拟合的 DTMC 提供进入当前事件的转移惊讶度。'], choice3Title:['Separate agent attribution from localization.','把智能体归因与步骤定位分开。'], choice3Body:["WHO sums content probability over an agent's events. WHEN combines event probability with transition surprise within the selected agent. Every signal can be inspected.",'WHO 按智能体汇总内容概率；WHEN 在选中的智能体内结合事件概率与转移惊讶度。每种信号都可被单独检查。'],
  evaluationLabel:['Evaluation evidence','评测证据'], evaluationTitle:['One final method. A defined evaluation.','一个最终方法，一组明确的评测。'], evaluationIntro:['The latest primary evaluation on Who&When Pro uses the same frozen core method, with text-only visible inputs.','最新的 Who&When Pro 主评测使用同一冻结核心方法，输入为可见文本轨迹。'], textOnly:['Text only','纯文本'], sdNote:['Mean ± sample standard deviation across 20 reported task-group holdouts. Overlapping splits describe variation across these runs, not independent replications.','数值为 20 次任务分组划分的均值 ± 样本标准差。测试划分存在重叠，标准差描述这些运行之间的变化。'], protocolTitle:['What was evaluated','评测对象与设置'], technicalDetails:['Inspect the frozen configuration & sources','查看冻结配置与数据来源'], footerNote:['A research project told through its method, recorded decisions and evaluation evidence.','通过方法流程、真实决策与评测证据，展示这项研究。'], downloadProject:['Save this project page ↗','保存作品集页面 ↗'], backTop:['Back to top ↑','回到顶部 ↑'],
  stageTitle1:['Turn visible events into comparable evidence.','把可见事件转成可比较的证据。'], stageBody1:['Each event becomes one frozen Agent / Action / State triple. The agent identifies the speaker, the action preserves task-affecting operations, and the state records the observed or claimed outcome.','每个事件对应一条冻结的 Agent／Action／State 三元组。Agent 标识发言者，Action 保留影响任务的操作，State 记录观察或声明的结果。'], stageNote1:['A reported claim remains a claim. Extraction does not independently verify it or consume the mistake annotation.','日志中的声明仍作为声明保存，提取过程不进行独立事实核查，也不读取错误标注。'], currentEvent:['Current recorded event','当前记录事件'],
  stageTitle2:['Build a vocabulary of behavior.','建立行为的抽象词汇。'], stageBody2:['Action and State use separate TF-IDF vocabularies. Joint PCA-32 and a diagonal GMM-16 map them to soft states. A group-weighted first-order DTMC estimates initial and transition frequencies.','Action 与 State 分别使用 TF-IDF 词表，经过联合 PCA-32 与对角 GMM-16 映射为软状态。一阶分组加权 DTMC 估计初始状态与状态转移频率。'], stageNote2:['These are fold-local state components. The scoring keeps soft assignments; the graph displays the most probable state for orientation.','这些状态在训练划分内形成。评分保留软状态分配，图中展示最可能的状态，帮助理解路径。'], stateAssignment:['Soft-state assignment at this event','当前事件的软状态分配'],
  stageTitle3:['Combine content evidence with incoming surprise.','结合内容证据与传入转移惊讶度。'], stageBody3:['S measures Error/Normal semantic frequency contrasts. R adds Action–State relational evidence. Their softmax gives content mass p. Incoming DTMC conformance c supplies surprise M = −log(c).','S 衡量错误／正常语义词频对比，R 加入动作—状态关系证据。二者的 softmax 给出内容质量 p；传入 DTMC 一致性 c 给出惊讶度 M = −log(c)。'], stageNote3:['At the first event, conformance uses π. Later events use the previous and current soft states with transition matrix T.','首个事件的一致性由初始分布 π 计算，后续事件由前后软状态与转移矩阵 T 共同计算。'], contentMass:['Content mass p','内容质量 p'], incomingSurprise:['Incoming surprise M','传入惊讶度 M'], weightedEvidence:['Evidence p × M','证据 p × M'],
  stageTitle4:['Select the agent, then locate its event.','先选择智能体，再定位它的事件。'], stageBody4:['WHO sums p across each agent’s events. WHEN maximizes p × M among events of the selected agent. The behavioral signal redistributes evidence within that agent and leaves WHO unchanged.','WHO 按智能体累加 p；WHEN 在选中智能体的事件中取 p × M 最大者。行为信号在该智能体内重新分配证据，不改变 WHO。'], stageNote4:['This recorded prediction matches the separately stored dataset annotation: source step 2, displayed as event 03.','这条记录的预测与单独保存的数据集标注一致：源步骤 2，对应界面事件 03。'], agentMass:['Content mass pooled by agent','按智能体汇总内容质量'], selectedEvent:['Selected event','选中事件'],
  contentSignal:['Content evidence','内容证据'], contentSignalDetail:['softmax of semantic + relation scores','语义分数与关系分数之和的 softmax'], surpriseSignal:['Incoming transition surprise','进入当前事件的转移惊讶度'], surpriseSignalDetail:['−log of soft-state conformance','软状态一致性的负对数'], finalSignal:['Final step mass','最终步骤质量'], finalSignalDetail:['agent mass × within-agent redistribution','智能体质量 × 智能体内重新分配'],
  explanationStart:['The initial event is scored through π. It receives little content evidence in this trace.','初始事件通过 π 评分，在这条轨迹中获得较低的内容证据。'], explanationPlan:['This event proposes a plan. Its agent receives little pooled content mass, so it is not selected for attribution.','该事件提出任务计划。这个智能体获得的汇总内容质量较低，因此没有被选中归因。'], explanationError:['The dataset annotates this actor-name claim as the first error. Its content mass is 47.26%; incoming surprise is 7.88. The final readout localizes event 03.','数据集将这条演员姓名声明标为首次错误。其内容质量为 47.26%，传入惊讶度为 7.88，最终读出定位到事件 03。'], explanationLater:['This later claim has slightly higher content mass (50.76%), but lower incoming surprise (1.70). Its final step mass is lower than event 03.','后续声明的内容质量略高（50.76%），但传入惊讶度较低（1.70），最终步骤质量低于事件 03。'], explanationVerify:['The verifier repeats the resulting first name. Agent-level pooling attributes this trace to the TV series expert.','验证者重复了得到的名字，智能体级汇总将这条轨迹归因于电视剧领域专家。'], explanationEnd:['This termination event has little content evidence. Termination alone does not determine the selected failure position.','终止事件的内容证据较少，终止信号本身不决定被定位的错误位置。'],
  whoDefinition:['Hit any annotated responsible agent, on multi-agent traces.','在多智能体轨迹中，命中任一已标注责任智能体。'], whenDefinition:['Exact annotated failure coordinate, on all test traces.','在全部测试轨迹中，精确命中已标注的故障坐标。'], poolSize:['Text traces in the pool','文本轨迹池'], taskGroups:['Task groups','任务分组'], holdouts:['Grouped holdouts','分组评测次数'], testRecords:['Distinct test traces','不同测试轨迹'], fitting:['Model fitting','模型拟合'], trainingOnly:['Isolated training side','隔离后的训练侧'], coordinateNote:['WHO excludes single-agent traces. WHEN uses dataset-native coordinates; Debate and DyLAN use round-level positions. Accepted alternative annotations are honored, and extraction failures remain in the evaluation denominator.','WHO 不包含单智能体轨迹。WHEN 使用数据集原生坐标，其中 Debate 与 DyLAN 使用轮次位置。评测接受已标注的可选答案，并保留提取失败样本的分母。'],
  graphTitle:['Observed path through the saved 16-state model','保存的 16 状态模型中的观测路径'], graphLegend:['Most probable states for six recorded events','六个记录事件的最可能状态'], graphConformance:['Incoming conformance','传入一致性'], annotated:['Annotated error','标注错误'], tvShort:['TV expert','电视剧专家'], languageShort:['Language expert','语言专家'], verifyShort:['Verifier','验证专家'], loadError:['The project evidence could not be loaded. Refresh the page or use the standalone project file.','未能载入项目证据，请刷新页面或使用独立作品集文件。']
};
const events = [
  {title:['Read the task','读取任务'],summary:['The TV series expert receives a task to identify a character’s first name through an actor-role connection.','电视剧领域专家收到任务，需要沿着演员与角色的关系找到角色名字。']},
  {title:['Propose a plan','提出计划'],summary:['The language expert proposes identifying the actor, finding the character, then extracting the first name.','语言专家提出先确定演员，再寻找角色，最后提取名字的计划。']},
  {title:['Identify the actor','确定演员'],summary:['The TV series expert claims the actor is Bartosz Opania. The dataset annotates this actor-name claim as the first mistake.','电视剧领域专家声称演员是 Bartosz Opania。数据集把这条演员姓名声明标为首次错误。']},
  {title:['Derive the character','推导角色'],summary:['Using that actor name, the same expert identifies Piotr Korzecki and returns the first name Piotr.','同一专家沿用这个演员姓名，得到角色 Piotr Korzecki，并返回名字 Piotr。']},
  {title:['Verify the answer','验证答案'],summary:['The verification expert repeats Piotr as the answer and declares the findings accurate.','验证专家重复 Piotr 作为答案，并声称结论准确。']},
  {title:['End the trace','结束轨迹'],summary:['The TV series expert sends a termination signal, completing the recorded trace.','电视剧领域专家发送终止信号，这条记录的轨迹结束。']}
];
const t = key => copy[key]?.[state.lang==='en'?0:1] ?? key;
const tr = pair => pair[state.lang==='en'?0:1];
const eventName = index => `${t('eventWord')} ${String(index+1).padStart(2,'0')}`;
function agentName(name, short=false) {
  return t(name.includes('PolishLanguage') ? (short?'languageShort':'languageShort') : name.includes('Verification') ? (short?'verifyShort':'verifyShort') : (short?'tvShort':'tvAgent'));
}
function tripleHTML(triple, className='triples') {
  return `<dl class="${className}">${['agent','action','state'].map(key=>`<div><dt>${key}</dt><dd>${escapeHTML(triple[key])}</dd></div>`).join('')}</dl>`;
}

function renderLanguage() {
  document.documentElement.lang=state.lang==='en'?'en':'zh-CN';
  document.querySelectorAll('[data-i18n]').forEach(element=>element.textContent=t(element.dataset.i18n));
  document.querySelectorAll('[data-i18n-html]').forEach(element=>element.innerHTML=t(element.dataset.i18nHtml));
  $('language-toggle').textContent=state.lang==='en'?'中文 ↗':'English ↗';
  $('language-toggle').setAttribute('aria-label',state.lang==='en'?'切换中文':'Switch to English');
  renderGraph();renderMethod();renderCase();renderEvaluation();
  document.dispatchEvent(new CustomEvent('portfolio:language'));
  updateReadingProgress();
}

function updateReadingProgress() {
  const root=document.documentElement, distance=root.scrollHeight-innerHeight;
  $('reading-progress').style.width=`${distance>0?Math.min(100,Math.max(0,scrollY/distance*100)):0}%`;
  let current=null;
  for(const id of ['method','case','evidence'])if($(id).getBoundingClientRect().top<=innerHeight*.35)current=id;
  document.querySelectorAll('.site-header nav a').forEach(link=>{
    if(link.getAttribute('href')===`#${current}`)link.setAttribute('aria-current','location');
    else link.removeAttribute('aria-current');
  });
}
let progressScheduled=false;
addEventListener('scroll',()=>{
  if(progressScheduled)return;
  progressScheduled=true;
  requestAnimationFrame(()=>{updateReadingProgress();progressScheduled=false;});
},{passive:true});
addEventListener('resize',updateReadingProgress);

function renderGraph() {
  const c=state.case, count=c.saved_model.responsibilities[state.event].length, width=510,height=342,center={x:255,y:155};
  const path=c.steps.map(row=>row.abstract_state_argmax);
  const active=path[state.event], previous=state.event>0?path[state.event-1]:null;
  const inspected=state.inspectedState ?? active;
  const coordinates=Array.from({length:count},(_,i)=>{
    const ring=i<8?1:0, k=i%8, radius=ring?120:74,angle=(k/8)*Math.PI*2-Math.PI/2+(ring?0:.22);
    return {x:center.x+radius*Math.cos(angle),y:center.y+radius*Math.sin(angle)};
  });
  for(const target of ['hero-graph','case-graph']){
    let svg=`<svg viewBox="65 0 380 ${height}" role="group" aria-label="${escapeHTML(t('graphTitle'))}"><title>${escapeHTML(t('graphTitle'))}</title><defs><marker id="${target}-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="var(--graph-path)"/></marker><marker id="${target}-error" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="var(--graph-error)"/></marker></defs>`;
    const edges=[];
    c.saved_model.transition_probability.forEach((row,from)=>{
      const to=row.reduce((best,value,i)=>i!==from && (best===null||value>row[best])?i:best,null);
      if(to!==null && !edges.some(edge=>edge[0]===to&&edge[1]===from)){edges.push([from,to]);const a=coordinates[from],b=coordinates[to];svg+=`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="var(--graph-muted)" stroke-width="1" opacity=".26"/>`;}
    });
    const observed=new Set();
    let selectedPath='', selectedColor='var(--graph-path)';
    for(let i=1;i<path.length;i++){
      const from=path[i-1],to=path[i],key=`${from}-${to}`;
      if(observed.has(key))continue;observed.add(key);
      const a=coordinates[from],b=coordinates[to], selected=from===previous&&to===active;
      const error=selected&&state.event===c.annotation.gold_step, color=error?'var(--graph-error)':'var(--graph-path)';
      let edgePath;
      if(from===to)edgePath=`M${a.x-10} ${a.y-8}C${a.x-40} ${a.y-43} ${a.x+40} ${a.y-43} ${a.x+10} ${a.y-8}`;
      else {const dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);edgePath=`M${a.x+dx/d*16} ${a.y+dy/d*16}L${b.x-dx/d*19} ${b.y-dy/d*19}`;}
      svg+=`<path d="${edgePath}" fill="none" class="${selected?'active-transition':''}" stroke="${color}" stroke-width="${selected?2.5:1.5}" opacity="${selected?1:.45}" marker-end="url(#${target}-${error?'error':'arrow'})"><title>T[S${from}, S${to}] = ${number(c.saved_model.transition_probability[from][to],6)}</title></path>`;
      if(selected){selectedPath=edgePath;selectedColor=color;}
    }
    if(selectedPath&&!reducedMotion.matches)svg+=`<circle class="graph-particle" r="3.5" fill="${selectedColor}" pointer-events="none"><animateMotion dur="1.8s" repeatCount="1" path="${selectedPath}"/></circle>`;
    coordinates.forEach((point,index)=>{
      const inPath=path.includes(index), selected=index===active;
      const fill=selected&&state.event===c.annotation.gold_step?'var(--graph-error)':selected?'var(--graph-active)':inPath?'var(--graph-path)':'var(--graph-idle)', text=selected||inPath?'var(--graph-text)':'var(--graph-label)';
      svg+=`<g class="state-node ${selected?'selected':''}" data-state="${index}" role="button" tabindex="${index===inspected?0:-1}" aria-pressed="${index===inspected}" aria-label="${escapeHTML(t('inspectState')+' S'+String(index).padStart(2,'0'))}"><circle cx="${point.x}" cy="${point.y}" r="22" fill="transparent" stroke="none" pointer-events="all"/>${selected?`<circle class="state-halo" cx="${point.x}" cy="${point.y}" r="28" fill="none" stroke="${fill}" opacity=".35"/>`:''}<circle cx="${point.x}" cy="${point.y}" r="${selected?20:inPath?16:12}" fill="${fill}"/><text x="${point.x}" y="${point.y+3}" text-anchor="middle" fill="${text}" font-size="${inPath?10:9}">${String(index).padStart(2,'0')}</text><title>S${String(index).padStart(2,'0')} · q=${number(c.saved_model.responsibilities[state.event][index],6)}</title></g>`;
    });
    svg+=`<rect x="194" y="133" width="122" height="54" rx="8" fill="var(--graph-bg)" opacity=".94" pointer-events="none"/><text x="${center.x}" y="${center.y-7}" text-anchor="middle" fill="var(--graph-label)" font-size="10" pointer-events="none">${t('graphConformance')}</text><text x="${center.x}" y="${center.y+16}" text-anchor="middle" fill="var(--graph-path)" font-size="19" pointer-events="none">${number(c.steps[state.event].incoming_conformance,5)}</text>`;
    svg+=`<text x="${width/2}" y="${height-34}" text-anchor="middle" fill="var(--graph-label)" font-size="10">${t('graphLegend')}</text>`;
    path.forEach((index,i)=>{const x=width/2-125+i*50;svg+=`<text x="${x}" y="${height-12}" text-anchor="middle" font-size="10" fill="${i===state.event?'var(--graph-error)':'var(--graph-label)'}">S${String(index).padStart(2,'0')}</text>${i<path.length-1?`<text x="${x+25}" y="${height-12}" text-anchor="middle" fill="var(--graph-muted)" font-size="9">→</text>`:''}`;});
    $(target).innerHTML=svg+'</svg>';
  }
  $('hero-state-label').textContent=previous===null?`π → S${String(active).padStart(2,'0')}`:`S${String(previous).padStart(2,'0')} → S${String(active).padStart(2,'0')}`;
  renderInspector(inspected,previous);
}

function renderInspector(index,previous) {
  const c=state.case, name=`S${String(index).padStart(2,'0')}`;
  const probability=previous===null?c.saved_model.initial_probability[index]:c.saved_model.transition_probability[previous][index];
  const from=previous===null?'π':`S${String(previous).padStart(2,'0')}`;
  const markup=`<h4>${t('inspectedState')} <strong>${name}</strong></h4><p>${eventName(state.event)}</p><dl><div><dt>${t('stateMembership')}</dt><dd>${number(c.saved_model.responsibilities[state.event][index]*100,3)}%</dd></div><div><dt>${previous===null?t('initialFrequency'):t('transitionFrequency')}<small>${from} → ${name}</small></dt><dd>${number(probability*100,3)}%</dd></div></dl><p class="inspector-note">${t('inspectorNote')}</p>`;
  $('case-inspector').innerHTML=markup;
  $('hero-inspector').hidden=!state.inspectManual;
  if(state.inspectManual)$('hero-inspector').innerHTML=markup;
}

function renderMethod() {
  const c=state.case, row=c.steps[state.event], triple=c.extracted_triples[state.event];
  document.querySelectorAll('[data-stage]').forEach(button=>{const selected=Number(button.dataset.stage)===state.stage;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;});
  const panel=$('method-panel');panel.setAttribute('aria-labelledby',`stage-tab-${state.stage}`);
  let detail='';
  if(state.stage===0)detail=`<h4>${t('currentEvent')} / ${eventName(state.event)}</h4>${tripleHTML(triple)}`;
  if(state.stage===1){
    const top=c.saved_model.responsibilities[state.event].map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p).slice(0,4);
    detail=`<div class="fit-sequence"><span>Action TF-IDF</span><span>State TF-IDF</span><b>→</b><span>PCA 32</span><b>→</b><span>GMM 16</span></div><h4 style="margin-top:24px">${t('stateAssignment')}</h4><div class="q-bars">${top.map(({p,i})=>`<div class="q-row"><span>S${String(i).padStart(2,'0')}</span><div class="q-track"><span style="width:${p*100}%"></span></div><span>${number(p*100,1)}%</span></div>`).join('')}</div>`;
  }
  if(state.stage===2)detail=`<h4>${eventName(state.event)}</h4><div class="formula">p = softmax(S + R)<br>M = −log(<span class="accent">${state.event===0?'q · π':'q<sub>prev</sub> · T · q'}</span>)</div><div class="formula-value"><div><small>${t('contentMass')}</small><strong>${number(row.content_probability*100)}%</strong></div><div><small>${t('incomingSurprise')}</small><strong>${number(row.dtmc_surprise)}</strong></div><div><small>${t('weightedEvidence')}</small><strong>${number(row.fused_evidence)}</strong></div></div>`;
  if(state.stage===3)detail=`<h4>${t('agentMass')}</h4>${Object.entries(c.agent_masses).sort((a,b)=>b[1]-a[1]).map(([name,p])=>`<div class="agent-row"><span>${agentName(name)}</span><strong>${number(p*100)}%</strong></div>`).join('')}<div class="formula">WHO = argmax<sub>a</sub> Σ p<sub>t</sub><br>WHEN = argmax<sub>t ∈ WHO</sub> (p<sub>t</sub> × M<sub>t</sub>)</div><div class="method-readout"><span>${t('selectedEvent')}</span><strong>${eventName(c.prediction.predicted_step)} · ${agentName(c.prediction.predicted_agent)}</strong></div>`;
  panel.innerHTML=`<div class="stage-copy"><h3>${t(`stageTitle${state.stage+1}`)}</h3><p>${t(`stageBody${state.stage+1}`)}</p><p class="detail-note">${t(`stageNote${state.stage+1}`)}</p></div><div class="stage-detail${reducedMotion.matches?'':' event-change'}">${detail}</div>`;
}

function renderCase() {
  const c=state.case, row=c.steps[state.event];
  $('trace-timeline').innerHTML=c.steps.map((step,i)=>`<button class="trace-event ${i===c.annotation.gold_step?'error-event':''}" data-event="${i}" aria-pressed="${i===state.event}" aria-label="${escapeHTML(eventName(i)+' '+tr(events[i].title))}"><span class="event-number">${String(i+1).padStart(2,'0')}${i===c.annotation.gold_step?' ↗':''}</span><strong>${tr(events[i].title)}</strong><small>${agentName(step.agent,true)}</small></button>`).join('');
  $('selected-event-label').textContent=`${eventName(state.event)} · ${state.lang==='en'?'source step':'源步骤'} ${state.event}`;
  $('selected-speaker').textContent=agentName(row.agent);
  $('selected-speaker').title=row.agent;
  $('event-summary').textContent=tr(events[state.event].summary);
  $('original-content').textContent=c.visible_steps[state.event].content;
  $('case-triple').innerHTML=['agent','action','state'].map(key=>`<div><dt>${key}</dt><dd>${escapeHTML(c.extracted_triples[state.event][key])}</dd></div>`).join('');
  const signals=[['contentSignal','contentSignalDetail',number(row.content_probability*100)+'%'],['surpriseSignal','surpriseSignalDetail',number(row.dtmc_surprise)],['finalSignal','finalSignalDetail',number(row.final_step_mass*100)+'%']];
  $('event-signals').innerHTML=signals.map(([label,detail,value])=>`<div class="signal-row"><div><span>${t(label)}</span><small>${t(detail)}</small></div><strong>${value}</strong></div>`).join('');
  $('event-explanation').textContent=t(['explanationStart','explanationPlan','explanationError','explanationLater','explanationVerify','explanationEnd'][state.event]);
  renderDistribution();
}

function renderDistribution() {
  const container=$('step-distribution'), initial=!container.firstElementChild;
  if(initial)container.innerHTML=`<div class="distribution-bars" role="group" aria-label="${escapeHTML(t('stepDistribution'))}">${state.case.steps.map((_,i)=>`<button type="button" class="distribution-cell" data-distribution="${i}"><span class="distribution-value"></span><span class="distribution-column" aria-hidden="true"><span style="height:0%"></span></span><span class="distribution-event">${String(i+1).padStart(2,'0')}</span></button>`).join('')}</div>`;
  $('evidence-content').setAttribute('aria-pressed',String(state.evidenceMode==='content'));
  $('evidence-fused').setAttribute('aria-pressed',String(state.evidenceMode==='final'));
  $('evidence-mode-note').textContent=t(state.evidenceMode==='content'?'contentViewNote':'finalViewNote');
  const heading=$('distribution-title');
  heading.dataset.i18n=state.evidenceMode==='content'?'contentViewHeading':'stepDistribution';
  heading.textContent=t(heading.dataset.i18n);
  container.firstElementChild.setAttribute('aria-label',heading.textContent);
  const updateBars=()=>state.case.steps.forEach((row,i)=>{
    const mass=state.evidenceMode==='content'?row.content_probability:row.final_step_mass;
    const button=container.querySelector(`[data-distribution="${i}"]`), value=`${number(mass*100)}%`;
    button.classList.toggle('selected',i===state.event);
    button.classList.toggle('error',i===state.case.annotation.gold_step);
    button.setAttribute('aria-pressed',String(i===state.event));
    button.setAttribute('aria-label',`${eventName(i)} · ${t(state.evidenceMode==='content'?'contentView':'finalView')} ${value}`);
    button.querySelector('.distribution-value').textContent=value;
    button.querySelector('.distribution-column > span').style.height=`${mass*100}%`;
  });
  if(initial&&!reducedMotion.matches)requestAnimationFrame(updateBars);else updateBars();
}

function selectEvent(index,source='manual') {
  if(!Number.isInteger(index)||index<0||index>=state.case.steps.length)return;
  const changed=index!==state.event;
  for(const id of ['event-explanation','case-inspector','hero-inspector'])$(id).setAttribute('aria-live',source==='playback'?'off':'polite');
  state.event=index;state.inspectedState=null;state.inspectManual=false;
  renderGraph();renderMethod();renderCase();
  if(changed&&!reducedMotion.matches)for(const pane of document.querySelectorAll('.trace-pane,.evidence-pane')){
    pane.classList.remove('event-change');requestAnimationFrame(()=>pane.classList.add('event-change'));
  }
  document.dispatchEvent(new CustomEvent('portfolio:event',{detail:{index,source}}));
}

function inspectGraph(index,target,focus=false) {
  state.inspectedState=index;state.inspectManual=true;
  renderGraph();
  document.dispatchEvent(new CustomEvent('portfolio:event',{detail:{index:state.event,source:'graph'}}));
  if(focus)$(target).querySelector(`[data-state="${index}"]`).focus({preventScroll:true});
}

function renderEvaluation() {
  const evaluation=state.data.evaluation,protocol=evaluation.protocol;
  $('evaluation-metrics').innerHTML=['who','when'].map(metric=>{
    const row=evaluation.metrics[metric],mean=row.mean*100,sd=row.sample_sd*100;
    return `<div class="metric-result"><div class="metric-result-top"><span>${metric.toUpperCase()}</span><strong>${number(mean)}% <small>± ${number(sd)}</small></strong></div><p>${t(metric+'Definition')}</p><div class="metric-track"><span style="width:${mean}%"></span><i class="metric-error" style="left:${Math.max(0,mean-sd)}%;width:${2*sd}%" aria-label="sample SD ${number(sd)} percentage points"></i></div><div class="metric-ticks"><span>0</span><span>50</span><span>100%</span></div></div>`;
  }).join('');
  const facts=[['poolSize',evaluation.total_traces.toLocaleString(state.lang)],['taskGroups',evaluation.task_groups.toLocaleString(state.lang)],['holdouts',String(protocol.splits)],['testRecords',protocol.unique_test_traces.toLocaleString(state.lang)],['fitting',t('trainingOnly')]];
  $('protocol-facts').innerHTML=facts.map(([key,value])=>`<div><dt>${t(key)}</dt><dd>${escapeHTML(value)}</dd></div>`).join('');
  $('evaluation-coordinate-note').textContent=t('coordinateNote');
  const config=state.data.project.config;
  $('method-config').textContent=`Canonical V2.1 · ${config.extraction.model}\nS: clause balance + single digits · α = ${config.semantic.alpha}\nR: Action–State relation · α = ${config.relation.alpha}\nSeparate TF-IDF → PCA ${config.abstraction.pca.dimensions} → diagonal GMM ${config.abstraction.gmm.states}\nFirst-order DTMC · α = ${config.dtmc.alpha}\n${state.data.project.readout.content_probability}\n${state.data.project.readout.who}\n${state.data.project.readout.incoming_conformance}\n${state.data.project.readout.transition_surprise}\n${state.data.project.readout.when}`;
  const provenance=state.data.provenance;
  $('source-evidence').replaceChildren();
  const text=document.createElement('p');text.textContent=provenance.file || provenance.source_files?.map(item=>typeof item==='string'?item:item.file).join('\n') || 'Frozen evaluation artifact';$('source-evidence').append(text);
  if(provenance.hash){const hash=document.createElement('p');hash.textContent=`SHA-256 ${provenance.hash}`;$('source-evidence').append(hash);}
}

document.querySelectorAll('[data-stage]').forEach(button=>{
  button.addEventListener('click',()=>{state.stage=Number(button.dataset.stage);renderMethod();});
  button.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();state.stage=event.key==='Home'?0:event.key==='End'?3:(state.stage+(event.key==='ArrowRight'?1:3))%4;renderMethod();$(`stage-tab-${state.stage}`).focus();
  });
});
$('trace-timeline').addEventListener('click',event=>{const button=event.target.closest('[data-event]');if(!button)return;selectEvent(Number(button.dataset.event));$('trace-timeline').querySelector(`[data-event="${state.event}"]`).focus({preventScroll:true});});
$('language-toggle').addEventListener('click',()=>{if(!state.case)return;state.lang=state.lang==='en'?'zh':'en';renderLanguage();});


$('step-distribution').addEventListener('click',event=>{const button=event.target.closest('[data-distribution]');if(!button)return;selectEvent(Number(button.dataset.distribution),'distribution');});
for(const [id,mode] of [['evidence-content','content'],['evidence-fused','final']])$(id).addEventListener('click',()=>{if(!state.case)return;state.evidenceMode=mode;renderDistribution();document.dispatchEvent(new CustomEvent('portfolio:event',{detail:{index:state.event,source:'manual'}}));});
for(const target of ['hero-graph','case-graph']){
  $(target).addEventListener('click',event=>{const node=event.target.closest('[data-state]');if(node)inspectGraph(Number(node.dataset.state),target,true);});
  $(target).addEventListener('keydown',event=>{
    const node=event.target.closest('[data-state]');if(!node)return;
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','Enter',' '].includes(event.key))return;
    event.preventDefault();
    const current=Number(node.dataset.state), count=state.case.saved_model.responsibilities[state.event].length;
    const next=event.key==='Home'?0:event.key==='End'?count-1:event.key==='ArrowLeft'||event.key==='ArrowUp'?(current+count-1)%count:event.key==='ArrowRight'||event.key==='ArrowDown'?(current+1)%count:current;
    inspectGraph(next,target,true);
  });
}
reducedMotion.addEventListener('change',()=>{if(state.case)renderGraph();});

(async()=>{
  try{
    if(window.PORTFOLIO_DATA && window.PORTFOLIO_CASE){state.data=window.PORTFOLIO_DATA;state.case=window.PORTFOLIO_CASE;}
    else{
      const version=window.PORTFOLIO_ASSET_VERSION?`?v=${encodeURIComponent(window.PORTFOLIO_ASSET_VERSION)}`:'';
      const responses=await Promise.all([fetch('./portfolio_data.json'+version),fetch('./portfolio_case.json'+version)]);
      if(responses.some(response=>!response.ok))throw new Error('Evidence unavailable');
      [state.data,state.case]=await Promise.all(responses.map(response=>response.json()));
    }
    renderLanguage();
    window.portfolioControls={selectEvent,getEvent:()=>state.event,getCount:()=>state.case.steps.length};
    document.dispatchEvent(new CustomEvent('portfolio:ready'));
  }catch(error){$('load-error').textContent=t('loadError');$('load-error').hidden=false;}
})();
