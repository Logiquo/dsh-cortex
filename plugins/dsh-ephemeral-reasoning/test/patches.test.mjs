import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import { agentEvents } from '@deepseek-ai/dsh-agent'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { AssistantStreamAccumulator, BlockAssembler, createAssistantMessage, createUserMessage, expandAssistantStream } from '@deepseek-ai/dsh-llm'
import { estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { interceptAppend, removedReasoningTokens } from '../lib/storage.js'
import { interceptMeter } from '../lib/meter.js'
import {TokenMeter} from '@deepseek-ai/dsh-token-meter'
import {SessionProjectionRegistry} from '@deepseek-ai/dsh-session-projection'
import * as plugin from '../lib/index.js'
const make=()=>Session.create(SessionId(crypto.randomUUID()))
function response(s,content=[{type:'reasoning',text:'private'},{type:'text',text:'answer'}]){
 return s.append('assistant/message',{turn:1,step:1,stream:[],message:createAssistantMessage({content,source:{provider:'deepseek-account',model:'test',replayState:{private:true}}})},{surfaceOp:'append'})
}
const HEADER={config:{provider:'deepseek-account',model:'test'}}
function envelope(s,header=HEADER){return s.append('request/header',{header,reason:'initial'})}
function settle(s,usage,content=[{type:'reasoning',text:'private-reasoning'},{type:'text',text:'answer'}]){
 return s.append('assistant/message',{turn:1,step:1,stream:[],...(usage===undefined?{}:{usage}),message:createAssistantMessage({content,source:{provider:'deepseek-account',model:'test'}})},{surfaceOp:'append'})
}
function prompt(s,text='next'){return s.append('user/message',createUserMessage({source:{kind:'user'},content:[{type:'text',text}]}),{surfaceOp:'append'})}
function raw(nodes){
 const surfaceTokens=nodes.reduce((total,node)=>total+node.tokens,0)
 return {logRevision:0,baseline:{kind:'estimated',tokens:surfaceTokens},surfaceDeltaTokens:0,totalTokens:surfaceTokens,surfaceTokens,nodes}
}
function fakeMeter(s){return {measure:()=>({logRevision:s.seq,baseline:{kind:'usage',tokens:60000,usage:{inputTokens:50000,outputTokens:10000,reasoningTokens:9000}},surfaceDeltaTokens:0,totalTokens:60000,surfaceTokens:133,nodes:[{seq:1,tokens:133,heuristicTokens:10}]}),contextPressure:()=>({pressureTokens:50000,contextWindow:100000})}}
const ANSWER=[{type:'text',text:'answer'}]
const visible=content=>estimateMessage(createAssistantMessage({content,source:{provider:'deepseek-account',model:'test'}}))
test('mid-turn: the provider total plus only the nodes appended after it',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100,cacheReadTokens:5000},ANSWER)
 const nodes=[{seq:a.seq,tokens:visible(ANSWER),heuristicTokens:visible(ANSWER)},{seq:a.seq+1,tokens:40,heuristicTokens:40}]
 const meter={measure:()=>raw(nodes),contextPressure:()=>({pressureTokens:50000,contextWindow:100000})}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.baseline.kind,'usage')
 assert.equal(m.baseline.tokens,6100,'billed prompt plus response, including its reasoning')
 assert.deepEqual(m.baseline.usage,{inputTokens:1000,outputTokens:100,cacheReadTokens:5000})
 assert.equal(m.surfaceDeltaTokens,40,'the open turn resends its reasoning, so nothing is dropped')
 assert.equal(m.totalTokens,6140)
 assert.equal(m.totalTokens,m.baseline.tokens+m.surfaceDeltaTokens)
 assert.equal(m.nodes,nodes,'route node prices are passed through untouched')
 assert.equal(meter.contextPressure(s).pressureTokens,50000)
 assert.equal(meter.measure(make()).baseline.kind,'estimated','a session with no settlement yet is priced from its surface')
 undo()
})
test('a user message closes the turn and removes exactly the reasoning it stops resending',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 const u=prompt(s)
 const reasoning=100-visible(ANSWER)
 const nodes=[{seq:a.seq,tokens:visible(ANSWER),heuristicTokens:visible(ANSWER)},{seq:u.seq,tokens:40,heuristicTokens:40}]
 const meter={measure:()=>raw(nodes)}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.baseline.kind,'usage')
 assert.equal(m.baseline.tokens,1100,'the recorded provider record is reported untouched')
 assert.equal(m.surfaceDeltaTokens,40-reasoning,'appended tail minus the dropped reasoning')
 assert.equal(m.totalTokens,1100+40-reasoning)
 assert.equal(m.totalTokens,m.baseline.tokens+m.surfaceDeltaTokens)
 undo()
})
test('every settled step of the closed turn contributes its reasoning exactly once',()=>{
 const s=make();envelope(s)
 const first=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 const second=settle(s,{inputTokens:1200,outputTokens:100},ANSWER)
 const u=prompt(s)
 const per=100-visible(ANSWER)
 const nodes=[{seq:first.seq,tokens:10,heuristicTokens:10},{seq:second.seq,tokens:10,heuristicTokens:10},{seq:u.seq,tokens:7,heuristicTokens:7}]
 const meter={measure:()=>raw(nodes)}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.totalTokens,1200+100+7-2*per,'both steps of the turn are dropped, not just the anchor')
 assert.equal(m.totalTokens,m.baseline.tokens+m.surfaceDeltaTokens)
 undo()
})
test('a settlement sanitation emptied contributes its whole output as dropped reasoning',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:50},[])
 const u=prompt(s)
 const nodes=[{seq:a.seq,tokens:0,heuristicTokens:0},{seq:u.seq,tokens:40,heuristicTokens:40}]
 const meter={measure:()=>raw(nodes)}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.totalTokens,1050+40-50,'an empty message derives to nothing, so no visible text is resent')
 undo()
})
test('an envelope change or a rewritten surface drops the anchor instead of anchoring stale history',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 const meter={measure:()=>raw([{seq:a.seq,tokens:60,heuristicTokens:60}])}
 const undo=interceptMeter(meter)
 assert.equal(meter.measure(s).baseline.kind,'usage')
 envelope(s,{config:{provider:'deepseek-account',model:'other'}})
 assert.equal(meter.measure(s).baseline.kind,'estimated','a new envelope bills different history')
 const other=make();envelope(other)
 const b=settle(other,{inputTokens:700,outputTokens:70},ANSWER)
 const replaced={measure:()=>raw([{seq:b.seq,tokens:70,heuristicTokens:70}])}
 const undoReplace=interceptMeter(replaced)
 assert.equal(replaced.measure(other).baseline.kind,'usage')
 other.append('user/message',createUserMessage({source:{kind:'user'},content:[{type:'text',text:'brief'}]}),{surfaceOp:{op:'replace',startSeq:b.seq,endSeq:b.seq},sourceEventSeqs:[b.seq]})
 assert.equal(replaced.measure(other).baseline.kind,'estimated','compaction rewrote the billed history')
 undo();undoReplace()
})
test('an effective header that was never logged still retires the anchor',()=>{
 const s=make();envelope(s)
 settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 const meter={measure:()=>raw([{seq:2,tokens:60,heuristicTokens:60}])}
 const undo=interceptMeter(meter)
 assert.equal(meter.measure(s).baseline.kind,'usage')
 const other={config:{provider:'deepseek-account',model:'other'}}
 assert.equal(meter.measure(s,other).baseline.kind,'estimated','the caller is billing a different route than the anchor recorded')
 assert.equal(meter.measure(s,HEADER).baseline.kind,'usage','the same route keeps the anchor')
 undo()
})
test('an identical envelope re-logged for a resumed series keeps the anchor',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('request/header',{header:HEADER,reason:'resume'})
 const meter={measure:()=>raw([{seq:a.seq,tokens:60,heuristicTokens:60}])}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.baseline.kind,'usage','the billed prompt still describes the same messages')
 assert.equal(m.baseline.tokens,1100)
 undo()
})
test('a settlement without usage keeps the previous anchor; its node stays in the unpriced tail',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:800,outputTokens:80},ANSWER)
 const silent=settle(s,undefined,ANSWER)
 const nodes=[{seq:a.seq,tokens:50,heuristicTokens:50},{seq:silent.seq,tokens:25,heuristicTokens:25}]
 const meter={measure:()=>raw(nodes)}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.baseline.kind,'usage')
 assert.equal(m.baseline.tokens,880)
 assert.equal(m.surfaceDeltaTokens,25,'the silent settlement is priced, not dropped')
 assert.equal(m.totalTokens,905)
 undo()
})
test('an unanchored session is priced from its current surface, not the host total',()=>{
 const s=make();envelope(s);prompt(s)
 const stale={...raw([{seq:2,tokens:12,heuristicTokens:12}]),baseline:{kind:'usage',tokens:60000},totalTokens:60000}
 const meter={measure:()=>stale}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.baseline.kind,'estimated')
 assert.equal(m.totalTokens,12,'current surface plus the (empty) tool schema cost')
 assert.equal(m.totalTokens,m.baseline.tokens+m.surfaceDeltaTokens)
 undo()
})
test('a recorded total whose history was rewritten is repriced, not reused',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('user/message',createUserMessage({source:{kind:'user'},content:[{type:'text',text:'brief'}]}),{surfaceOp:{op:'replace',startSeq:a.seq,endSeq:a.seq},sourceEventSeqs:[a.seq]})
 const stale={...raw([{seq:a.seq,tokens:10,heuristicTokens:10}]),baseline:{kind:'usage',tokens:9000},totalTokens:9000}
 const meter={measure:()=>stale}
 const undo=interceptMeter(meter)
 const m=meter.measure(s)
 assert.equal(m.baseline.kind,'estimated','a pre-compaction total cannot price the post-compaction surface')
 assert.equal(m.totalTokens,10)
 assert.equal(m.totalTokens,m.baseline.tokens+m.surfaceDeltaTokens)
 undo()
})
test('append and meter wrappers deactivate beneath later extensions without overwriting them',()=>{
 const s=make();const undo=interceptAppend(s);const inner=s.append;const outer=function(...args){return inner.apply(this,args)};s.append=outer
 response(s);assert.equal(s.deriveMessages()[0].content.length,1);undo();assert.equal(s.append,outer)
 response(s);assert.equal(s.deriveMessages().at(-1).content.length,2)
 const meter=fakeMeter(s);const unm=interceptMeter(meter);const inside=meter.measure;meter.measure=function(...args){return inside.apply(this,args)};unm();assert.equal(meter.measure(s).baseline.kind,'usage')
})
async function setup(existing=[]){
 const ctx=new Context();const agents=existing.map(s=>({id:s.id,session:s,ctx}));
 const projections=ctx.plugin(SessionProjectionRegistry);await projections.await();
 const meterFiber=ctx.plugin(TokenMeter);await meterFiber.await();const meter=ctx.tokenMeter
 ctx.provide('agents',{list:()=>[...agents]});const fiber=ctx.plugin(plugin,{});await fiber.await();return {ctx,fiber,meter}
}
test('existing agents attach for new writes; new/resumed agents attach; dispose restores append',async()=>{
 const s=make();response(s);const base=s.append;const {ctx,fiber}=await setup([s]);response(s)
 assert.equal(s.deriveMessages()[0].content.length,2,'existing history is not migrated')
 assert.equal(s.deriveMessages().at(-1).content.length,1)
 const next=make();const agent={id:next.id,session:next,ctx};await agentEvents(ctx,agent).serial('agent/created',{source:'resume'});response(next);assert.equal(next.deriveMessages()[0].content.length,1)
 agentEvents(ctx,agent).emit('agent/disposed',{});response(next);assert.equal(next.deriveMessages().at(-1).content.length,2)
 await fiber.dispose();assert.equal(s.append,base)
 const mounted=ctx.plugin(plugin,{});await mounted.await();response(s);assert.equal(s.deriveMessages().at(-1).content.length,1);await mounted.dispose()
})
test('all provider/model routes pass admission; disabled plugin leaves writes untouched',async()=>{
 const s=make();const {ctx,fiber}=await setup([s]);const agent={id:s.id,session:s,ctx};let dispatched=false
 for (const [provider,model] of [['deepseek-account','deepseek-flash'],['anthropic','claude-test'],['openai','reasoning-test'],['custom-route','custom-model']]) {
  const decision=await agentEvents(ctx,agent).waterfall('agent/pre-step',{messages:[],turn:1,step:1,signal:new AbortController().signal,modelSelection:{provider,model}},async()=>{dispatched=true;return {kind:'enter',messages:[]}})
  assert.equal(decision.kind,'enter');assert.equal(dispatched,true);dispatched=false
  const event=s.append('assistant/message',{turn:1,step:1,stream:[],message:createAssistantMessage({source:{provider,model,replayState:{private:true}},content:[{type:'reasoning',text:'private'},{type:'text',text:'answer'}]})},{surfaceOp:'append'})
  assert.deepEqual(event.data.message.source,{kind:'model',provider,model})
  assert.deepEqual(event.data.message.content,[{type:'text',text:'answer'}])
 }
 await fiber.dispose();const disabled=ctx.plugin(plugin,{enabled:false});await disabled.await();response(s);assert.equal(s.deriveMessages().at(-1).content.length,2);await disabled.dispose()
})
test('subagents are sanitized like main sessions; unload restores normal settlement',()=>{
 const s=make()
 const id=SessionId(crypto.randomUUID())
 const child=Session.create(id,[],{...s.header,id,origin:'subagent'})
 const undo=interceptAppend(child)
 response(child)
 assert.equal(child.deriveMessages()[0].content.length,1,'subagent reasoning must not be retained')
 undo()
 response(child)
 assert.equal(child.deriveMessages().at(-1).content.length,2)
 const dispose=interceptAppend(s);dispose();response(s);assert.equal(s.deriveMessages()[0].content.length,2)
})
test('the occupancy ring reports the same corrected total as the meter',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 envelope(s)
 const stepStart=(turn,step)=>s.append('step/start',{turn,step})
 const stepEnd=(turn,step)=>s.append('step/end',{turn,step})
 prompt(s,'first')
 stepStart(1,1)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 stepEnd(1,1)
 const ring=()=>ctx.sessionProjections.snapshot(s,['contextPressure']).values.contextPressure
 const midMeter=meter.measure(s).totalTokens
 assert.equal(ring().projectedTokens,midMeter,'ring and meter disagree while the turn is open')
 prompt(s,'second')
 const endMeter=meter.measure(s).totalTokens
 assert.equal(ring().projectedTokens,endMeter,'ring and meter disagree after the turn closed')
 assert.ok(endMeter < midMeter,'closing the turn must drop its reasoning')
 const nextPrompt=createUserMessage({source:{kind:'user'},content:[{type:'text',text:'second'}]})
 assert.equal(midMeter + estimateMessage(nextPrompt) - endMeter, 100 - visible(ANSWER), 'exactly the reasoning is dropped, not the visible answer')
 await fiber.dispose()
})

test('subagent occupancy is corrected exactly like a main session',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 const id=SessionId(crypto.randomUUID())
 const child=Session.create(id,[],{...s.header,id,origin:'subagent'})
 envelope(child)
 prompt(child,'first')
 child.append('step/start',{turn:1,step:1})
 settle(child,{inputTokens:900,outputTokens:90},ANSWER)
 child.append('step/end',{turn:1,step:1})
 const ring=()=>ctx.sessionProjections.snapshot(child,['contextPressure']).values.contextPressure
 assert.equal(ring().projectedTokens,meter.measure(child).totalTokens,'subagent ring and meter must agree')
 prompt(child,'second')
 const closed=meter.measure(child).totalTokens
 assert.equal(ring().projectedTokens,closed,'subagent ring must drop the closed turn reasoning')
 assert.ok(closed < meter.measure(child).baseline.tokens + 1,'the dropped reasoning came off the total')
 await fiber.dispose()
})
test('a projection definition installed later is not clobbered on unload',async()=>{
 const {ctx,fiber}=await setup([])
 const registration=ctx.sessionProjections.registrations.get('contextPressure')
 const ours=registration.def
 const later={...ours}
 registration.def=later
 await fiber.dispose()
 assert.equal(registration.def,later,'unload must not overwrite a definition installed after this one')
})

/* -------------------------------------------------------------------------- *
 * Regressions for the defects found in review
 * -------------------------------------------------------------------------- */

test('a second user message in the same loop keeps the closed turn reasoning dropped',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 envelope(s);prompt(s,'first')
 s.append('step/start',{turn:1,step:1})
 settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('step/end',{turn:1,step:1})
 const ring=()=>ctx.sessionProjections.snapshot(s,['contextPressure']).values.contextPressure.projectedTokens
 prompt(s,'second')
 assert.equal(ring(),meter.measure(s).totalTokens,'ring and meter must agree after the first close')
 const closed=ring()
 for (const text of ['third','fourth','fifth']) {
  prompt(s,text)
  assert.equal(ring(),meter.measure(s).totalTokens,`ring and meter must agree after "${text}"`)
 }
 assert.ok(closed < ring(),'the extra prompts add their own tokens and never resurrect the dropped reasoning')
 assert.equal(ring()-closed,['third','fourth','fifth'].reduce((total,text)=>total+estimateMessage(createUserMessage({source:{kind:'user'},content:[{type:'text',text}]})),0),'only the appended prompts move the total')
 await fiber.dispose()
})
test('a replace mid-turn keeps the open turn reasoning and the two surfaces agree at close',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 envelope(s);prompt(s,'first')
 s.append('step/start',{turn:1,step:1})
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('step/end',{turn:1,step:1})
 const tool=s.append('tool/result',{message:{role:'tool',source:{kind:'tool',callId:'c1'},toolCallId:'c1',content:[{type:'text',text:'x'.repeat(400)}]}},{surfaceOp:'append'})
 s.append('compaction/prune',{shadowedRange:{start:tool.seq,end:tool.seq},shadowedSeqs:[tool.seq],shadowedTokenCount:100})
 s.append('tool/result',{message:{role:'tool',source:{kind:'tool',callId:'c1'},toolCallId:'c1',content:[{type:'text',text:'pruned'}]}},{surfaceOp:{op:'replace',startSeq:tool.seq,endSeq:tool.seq},sourceEventSeqs:[tool.seq]})
 s.append('step/start',{turn:1,step:2})
 s.append('assistant/message',{turn:1,step:2,stream:[],usage:{inputTokens:1100,outputTokens:120},message:createAssistantMessage({content:ANSWER,source:{provider:'deepseek-account',model:'test'}})},{surfaceOp:'append'})
 s.append('step/end',{turn:1,step:2})
 const ring=()=>ctx.sessionProjections.snapshot(s,['contextPressure']).values.contextPressure.projectedTokens
 prompt(s,'close')
 assert.equal(ring(),meter.measure(s).totalTokens,'a replaced tool result must not erase the open turn reasoning')
 assert.ok(ring() < 1220,'the closed turn reasoning came off the total')
 await fiber.dispose()
})
test('dense scripts are not misread as reasoning, and no reasoning is inferred from a settlement that had none',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 const undo=interceptAppend(s)
 const cjk='这是一个需要仔细分析的问题，因为其中涉及多个相互依赖的约束条件。'.repeat(20)
 envelope(s);prompt(s,'ask')
 s.append('step/start',{turn:1,step:1})
 const clean=s.append('assistant/message',{turn:1,step:1,stream:[],usage:{inputTokens:500,outputTokens:cjk.length},message:createAssistantMessage({content:[{type:'text',text:cjk}],source:{provider:'deepseek-account',model:'test'}})},{surfaceOp:'append'})
 assert.equal(removedReasoningTokens(clean.data),0,'a settlement with no reasoning records zero, not an estimate')
 s.append('step/end',{turn:1,step:1})
 const ring=()=>ctx.sessionProjections.snapshot(s,['contextPressure']).values.contextPressure.projectedTokens
 prompt(s,'next')
 assert.equal(ring(),meter.measure(s).totalTokens,'the two surfaces must agree on dense text')
 assert.equal(meter.measure(s).totalTokens,500+cjk.length+estimateMessage(createUserMessage({source:{kind:'user'},content:[{type:'text',text:'next'}]})),'no reasoning is invented, so nothing is subtracted')
 undo();await fiber.dispose()
})
test('a settlement without a stream field is admitted unchanged',()=>{
 const s=make();const undo=interceptAppend(s)
 let event
 assert.doesNotThrow(()=>{event=s.append('assistant/message',{turn:1,step:1,message:createAssistantMessage({content:[{type:'text',text:'hi'}],source:{provider:'p',model:'m'}})},{surfaceOp:'append'})})
 assert.equal(event.data.stream,undefined,'no stream field is invented')
 assert.equal(s.deriveMessages().at(-1).content.length,1)
 undo()
})
test('replay state is kept when sanitation removed no reasoning',()=>{
 const s=make();const undo=interceptAppend(s)
 const state={response:{kind:'deepseek-messages',version:1,model:'m'},blocks:[{type:'text'}]}
 const event=s.append('assistant/message',{turn:1,step:1,stream:[],message:createAssistantMessage({content:[{type:'text',text:'answer'}],source:{provider:'p',model:'m',replayState:state}})},{surfaceOp:'append'})
 assert.deepEqual(event.data.message.source.replayState,state,'a text-only settlement keeps its provider replay state')
 undo()
 const s2=make();const undo2=interceptAppend(s2)
 const stripped=s2.append('assistant/message',{turn:1,step:1,stream:[],message:createAssistantMessage({content:[{type:'reasoning',text:'private'},{type:'text',text:'answer'}],source:{provider:'p',model:'m',replayState:state}})},{surfaceOp:'append'})
 assert.equal(stripped.data.message.source.replayState,undefined,'a sanitized settlement drops the envelope its blocks no longer match')
 undo2()
})
test('the ring does not anchor before a request header is logged',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 prompt(s,'first')
 s.append('step/start',{turn:1,step:1})
 settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('step/end',{turn:1,step:1})
 const ring=()=>ctx.sessionProjections.snapshot(s,['contextPressure']).values.contextPressure.projectedTokens
 assert.equal(ring(),meter.measure(s).totalTokens,'the ring must not anchor where the meter refuses to')
 await fiber.dispose()
})
test('after a replacement the ring prices the surface instead of the stale pre-compaction sample',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 envelope(s);prompt(s,'first')
 s.append('step/start',{turn:1,step:1})
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('step/end',{turn:1,step:1})
 prompt(s,'second')
 s.append('user/message',createUserMessage({source:{kind:'user'},content:[{type:'text',text:'brief'}]}),{surfaceOp:{op:'replace',startSeq:a.seq,endSeq:a.seq},sourceEventSeqs:[a.seq]})
 const ring=()=>ctx.sessionProjections.snapshot(s,['contextPressure']).values.contextPressure.projectedTokens
 assert.equal(ring(),meter.measure(s).totalTokens,'both surfaces must fall back to the same reprice')
 assert.ok(ring()<1000,'the pre-compaction pressure must not survive the rewrite')
 await fiber.dispose()
})
test('a session that cannot be wrapped does not fail agent creation',async()=>{
 const {ctx,fiber}=await setup([])
 const hostile=make()
 Object.defineProperty(hostile,'append',{value:hostile.append,writable:false,configurable:false})
 assert.throws(()=>hostile.append=function(){},/read only|Cannot assign/i)
 const agent={id:hostile.id,session:hostile,ctx}
 assert.doesNotThrow(()=>agentEvents(ctx,agent).emit('agent/created',{source:'new'}))
 assert.doesNotThrow(()=>response(hostile),'the unwrapped session still accepts writes under host behaviour')
 await fiber.dispose()
})
test('a refused occupancy override leaves no meter patch and no owner marker behind',async()=>{
 const ctx=new Context()
 const projections=ctx.plugin(SessionProjectionRegistry);await projections.await()
 const meterFiber=ctx.plugin(TokenMeter);await meterFiber.await()
 const registration=ctx.sessionProjections.registrations.get('contextPressure')
 const real=registration.def
 registration.def={...real,stateVersion:4}
 ctx.provide('agents',{list:()=>[]})
 const before=Object.getOwnPropertyDescriptor(ctx.tokenMeter,'measure')
 const fiber=ctx.plugin(plugin,{})
 await assert.rejects(fiber.await(),/stateVersion 4/)
 assert.deepEqual(Object.getOwnPropertyDescriptor(ctx.tokenMeter,'measure'),before,'the meter must not stay patched')
 assert.equal(Reflect.get(ctx.tokenMeter,Symbol.for('dsh-ephemeral-reasoning.owner')),undefined,'the owner marker must be released')
 registration.def=real
 const retry=ctx.plugin(plugin,{})
 await assert.doesNotReject(retry.await(),undefined,'the plugin must be mountable again after a refusal')
 await retry.dispose();await meterFiber.dispose();await projections.dispose()
})
test('a settlement that carried no reasoning still reassembles to its own message',()=>{
 // dsh's persistence verifies every stored settlement against its embedded
 // stream: content, usage, and replay state must all agree. Dropping the replay
 // envelope from only one of the two breaks a log written with no reasoning.
 const replayState={response:{kind:'deepseek-messages',version:1,model:'test'},blocks:[{type:'text'}]}
 const chunks=[
  {type:'block-start',index:0,blockType:'text'},
  {type:'text-delta',index:0,text:'answer'},
  {type:'block-end',index:0,block:{type:'text',text:'answer'}},
  {type:'usage',usage:{inputTokens:10,outputTokens:5}},
  {type:'finish',reason:{kind:'stop'},replayState},
 ]
 const accumulator=new AssistantStreamAccumulator()
 const live=new BlockAssembler()
 for(const chunk of chunks){const timed=accumulator.push({time:1,chunk});live.push(timed.chunk)}
 const s=make();const undo=interceptAppend(s)
 const event=s.append('assistant/message',{turn:1,step:1,stream:accumulator.snapshot(),usage:live.usage,message:createAssistantMessage({content:live.blocks(),source:{provider:'deepseek-account',model:'test',replayState}})},{surfaceOp:'append'})
 const restored=new BlockAssembler()
 for(const member of expandAssistantStream(event.data.stream)) restored.push(member.chunk)
 assert.deepEqual(restored.blocks(),event.data.message.content,'content must reassemble')
 assert.deepEqual(restored.usage,event.data.usage,'usage must reassemble')
 assert.deepEqual(restored.replayState,event.data.message.source.replayState,'replay state must reassemble; the two must never disagree')
 assert.deepEqual(restored.replayState,replayState,'a settlement with no reasoning keeps the envelope it was written with')
 undo()
})
test('a session created while the occupancy unit is registering is still sanitized',async()=>{
 const host=new Context()
 const hostProjections=host.plugin(SessionProjectionRegistry);await hostProjections.await()
 const hostMeter=host.plugin(TokenMeter);await hostMeter.await()
 const ctx=new Context();const agents=[]
 // The real registry answers with a fresh array, so a list taken before the
 // wait cannot see an agent created during it.
 ctx.provide('agents',{list:()=>[...agents]})
 ctx.provide('tokenMeter',host.tokenMeter)
 // A registry whose unit appears only after the mount has begun, the way the
 // host's own asynchronous init registers it.
 const registrations=new Map()
 ctx.provide('sessionProjections',{registrations})
 const fiber=ctx.plugin(plugin,{})
 await new Promise(resolve=>setTimeout(resolve,40))
 const late=make()
 agents.push({id:late.id,session:late,ctx})
 registrations.set('contextPressure',{
  def:{key:'contextPressure',stateVersion:5,stateSchema:{parse:value=>value},init:()=>({surfaceTokens:0}),apply:state=>state,wire:{viewSchema:{parse:value=>value},view:()=>({}),summary:false}},
  cells:new WeakMap(),refs:1,
 })
 await fiber.await()
 response(late)
 assert.equal(late.deriveMessages().at(-1).content.length,1,'a session created while the mount waited must be sanitized')
 await fiber.dispose();await hostMeter.dispose();await hostProjections.dispose()
})
test('a corrupt occupancy row cannot poison the fold',async()=>{
 const {ctx,fiber}=await setup([])
 const parse=ctx.sessionProjections.registrations.get('contextPressure').def.stateSchema.parse
 const base={surfaceTokens:10,appendedTokens:4,turnReasoning:8,droppedReasoning:2}
 assert.deepEqual(parse(base),base,'a row this plugin wrote round-trips')
 assert.equal(parse({...base,claim:'nope'}).claim,undefined,'a malformed pending claim is dropped, not handed to the host fold')
 assert.equal(parse({...base,claim:{start:1,end:2,tokens:'x'}}).claim,undefined,'a partially numeric claim is dropped')
 assert.equal(parse({...base,envelope:'nope',billedTotal:100}).billedTotal,undefined,'a billed total without a usable envelope is dropped')
 assert.equal(parse({...base,envelope:{config:{provider:'p',model:'m'}},billedTotal:100}).billedTotal,100,'a usable anchor survives')
 assert.throws(()=>parse({surfaceTokens:1,appendedTokens:0,turnReasoning:0}),/droppedReasoning/,'a missing required tally still fails loudly')
 await fiber.dispose()
})
test('an attempt gives up its replay envelope only when sanitation removed blocks',()=>{
 const s=make();const undo=interceptAppend(s)
 const pack=(chunks)=>{const accumulator=new AssistantStreamAccumulator();for(const chunk of chunks)accumulator.push({time:1,chunk});return accumulator.snapshot()}
 const replayState={response:{kind:'deepseek-messages',version:1,model:'test'},blocks:[{type:'text'}]}
 const finish={type:'finish',reason:{kind:'stop'},replayState}
 const plain=s.append('assistant/attempt',{turn:1,step:1,stream:pack([{type:'block-start',index:0,blockType:'text'},{type:'text-delta',index:0,text:'partial'},finish])})
 const envelope=plain.data.stream.find(record=>record.type==='chunk'&&record.chunk.type==='finish')
 assert.deepEqual(envelope.chunk.replayState,replayState,'an attempt with no reasoning keeps the envelope it was written with')
 const reasoned=s.append('assistant/attempt',{turn:1,step:1,stream:pack([{type:'block-start',index:0,blockType:'reasoning'},{type:'reasoning-delta',index:0,text:'private'},finish])})
 const dropped=reasoned.data.stream.find(record=>record.type==='chunk'&&record.chunk.type==='finish')
 assert.equal(dropped.chunk.replayState,undefined,'an attempt that lost blocks drops the envelope describing them')
 undo()
})
test('a malformed settlement is rejected by the host, not by the sanitizer',()=>{
 // The host names the caller's mistake; a sanitizer that dereferences the
 // payload first replaces that with a TypeError that says nothing.
 const s=make();const undo=interceptAppend(s)
 assert.throws(()=>s.append('assistant/message'),/non-JSON-serializable/,'no payload')
 assert.throws(()=>s.append('assistant/attempt'),/non-JSON-serializable/,'an attempt without a payload is the host\'s rejection')
 const rejections=[
  [{turn:1,step:1},/surfaceOp marker/,'no message'],
  [{turn:1,step:1,stream:[],message:{role:'assistant',content:'text',source:{kind:'model',provider:'p',model:'m'}}},/surfaceOp marker/,'non-array content'],
 ]
 for(const [payload,pattern,label] of rejections){
  assert.throws(()=>s.append('assistant/message',payload),error=>error instanceof Error&&!error.message.startsWith('Cannot read')&&pattern.test(error.message),label)
 }
 undo()
})
test('a finished turn stops resending its reasoning, not only the next prompt',async()=>{
 // Between `turn/end` and the next input the answer is on screen and the
 // composer ring is being read, so the number must already exclude reasoning
 // the next request will not bill.
 const s=make();const {ctx,fiber,meter}=await setup([])
 envelope(s);prompt(s,'first')
 s.append('step/start',{turn:1,step:1})
 settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('step/end',{turn:1,step:1})
 const ring=()=>ctx.sessionProjections.snapshot(s,['contextPressure']).values.contextPressure.projectedTokens
 const open=meter.measure(s).totalTokens
 assert.equal(ring(),open,'the turn is still open before turn/end')
 s.append('turn/end',{turn:1,reason:{kind:'completed'}})
 const closed=meter.measure(s).totalTokens
 assert.equal(open-closed,100-visible(ANSWER),'exactly the closed turn reasoning is dropped at turn/end')
 assert.equal(ring(),closed,'the ring must drop it at the same moment the meter does')
 prompt(s,'second')
 assert.equal(meter.measure(s).totalTokens,closed+estimateMessage(createUserMessage({source:{kind:'user'},content:[{type:'text',text:'second'}]})),'the next prompt adds its own price and the drop is not applied twice')
 assert.equal(ring(),meter.measure(s).totalTokens,'both surfaces agree after the next prompt lands')
 await fiber.dispose()
})
test('every settled step of a turn closed by turn/end is dropped once',async()=>{
 const s=make();const {ctx,fiber,meter}=await setup([])
 envelope(s);prompt(s,'first')
 s.append('step/start',{turn:1,step:1})
 settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('step/end',{turn:1,step:1})
 s.append('step/start',{turn:1,step:2})
 s.append('assistant/message',{turn:1,step:2,stream:[],usage:{inputTokens:1200,outputTokens:100},message:createAssistantMessage({content:ANSWER,source:{provider:'deepseek-account',model:'test'}})},{surfaceOp:'append'})
 s.append('step/end',{turn:1,step:2})
 const open=meter.measure(s).totalTokens
 s.append('turn/end',{turn:1,reason:{kind:'completed'}})
 const closed=meter.measure(s).totalTokens
 assert.equal(open-closed,2*(100-visible(ANSWER)),'both settled steps of the turn are dropped')
 s.append('turn/start',{turn:2})
 s.append('step/start',{turn:2,step:1})
 s.append('assistant/message',{turn:2,step:1,stream:[],usage:{inputTokens:1500,outputTokens:60},message:createAssistantMessage({content:ANSWER,source:{provider:'deepseek-account',model:'test'}})},{surfaceOp:'append'})
 s.append('step/end',{turn:2,step:1})
 assert.equal(meter.measure(s).baseline.tokens,1560,'a new settlement anchors on the provider count again, carrying no stale drop')
 await fiber.dispose()
})
test('the occupancy unit carries a version that discards rows from the old fold',async()=>{
 const {ctx,fiber}=await setup([])
 assert.equal(ctx.sessionProjections.registrations.get('contextPressure').def.stateVersion,8,'a persisted version-7 row must refold rather than seed the corrected fold')
 await fiber.dispose()
})
