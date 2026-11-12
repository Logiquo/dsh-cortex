import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import { agentEvents } from '@deepseek-ai/dsh-agent'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { interceptAppend } from '../lib/storage.js'
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
 const undo=interceptMeter(meter,x=>x===s)
 const m=meter.measure(s)
 assert.equal(m.baseline.kind,'usage')
 assert.equal(m.baseline.tokens,6100,'billed prompt plus response, including its reasoning')
 assert.deepEqual(m.baseline.usage,{inputTokens:1000,outputTokens:100,cacheReadTokens:5000})
 assert.equal(m.surfaceDeltaTokens,40,'the open turn resends its reasoning, so nothing is dropped')
 assert.equal(m.totalTokens,6140)
 assert.equal(m.totalTokens,m.baseline.tokens+m.surfaceDeltaTokens)
 assert.equal(m.nodes,nodes,'route node prices are passed through untouched')
 assert.equal(meter.contextPressure(s).pressureTokens,50000)
 assert.equal(meter.measure(make()).baseline.kind,'estimated','unowned sessions are measured by the host')
 undo()
})
test('a user message closes the turn and removes exactly the reasoning it stops resending',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 const u=prompt(s)
 const reasoning=100-visible(ANSWER)
 const nodes=[{seq:a.seq,tokens:visible(ANSWER),heuristicTokens:visible(ANSWER)},{seq:u.seq,tokens:40,heuristicTokens:40}]
 const meter={measure:()=>raw(nodes)}
 const undo=interceptMeter(meter,()=>true)
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
 const undo=interceptMeter(meter,()=>true)
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
 const undo=interceptMeter(meter,()=>true)
 const m=meter.measure(s)
 assert.equal(m.totalTokens,1050+40-50,'an empty message derives to nothing, so no visible text is resent')
 undo()
})
test('an envelope change or a rewritten surface drops the anchor instead of anchoring stale history',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 const meter={measure:()=>raw([{seq:a.seq,tokens:60,heuristicTokens:60}])}
 const undo=interceptMeter(meter,()=>true)
 assert.equal(meter.measure(s).baseline.kind,'usage')
 envelope(s,{config:{provider:'deepseek-account',model:'other'}})
 assert.equal(meter.measure(s).baseline.kind,'estimated','a new envelope bills different history')
 const other=make();envelope(other)
 const b=settle(other,{inputTokens:700,outputTokens:70},ANSWER)
 const replaced={measure:()=>raw([{seq:b.seq,tokens:70,heuristicTokens:70}])}
 const undoReplace=interceptMeter(replaced,()=>true)
 assert.equal(replaced.measure(other).baseline.kind,'usage')
 other.append('user/message',createUserMessage({source:{kind:'user'},content:[{type:'text',text:'brief'}]}),{surfaceOp:{op:'replace',startSeq:b.seq,endSeq:b.seq},sourceEventSeqs:[b.seq]})
 assert.equal(replaced.measure(other).baseline.kind,'estimated','compaction rewrote the billed history')
 undo();undoReplace()
})
test('an identical envelope re-logged for a resumed series keeps the anchor',()=>{
 const s=make();envelope(s)
 const a=settle(s,{inputTokens:1000,outputTokens:100},ANSWER)
 s.append('request/header',{header:HEADER,reason:'resume'})
 const meter={measure:()=>raw([{seq:a.seq,tokens:60,heuristicTokens:60}])}
 const undo=interceptMeter(meter,()=>true)
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
 const undo=interceptMeter(meter,()=>true)
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
 const undo=interceptMeter(meter,()=>true)
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
 const undo=interceptMeter(meter,()=>true)
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
 const meter=fakeMeter(s);const unm=interceptMeter(meter,()=>true);const inside=meter.measure;meter.measure=function(...args){return inside.apply(this,args)};unm();assert.equal(meter.measure(s).baseline.kind,'usage')
})
async function setup(existing=[]){
 const ctx=new Context();const agents=existing.map(s=>({id:s.id,session:s,ctx}));
 const projections=ctx.plugin(SessionProjectionRegistry);await projections.await();
 const meterFiber=ctx.plugin(TokenMeter);await meterFiber.await();const meter=ctx.tokenMeter
 ctx.provide('agents',{list:()=>agents});const fiber=ctx.plugin(plugin,{});await fiber.await();return {ctx,fiber,meter}
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
