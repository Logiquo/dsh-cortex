import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import { agentEvents } from '@deepseek-ai/dsh-agent'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { createAssistantMessage } from '@deepseek-ai/dsh-llm'
import { interceptAppend } from '../lib/storage.js'
import { interceptMeter } from '../lib/meter.js'
import {TokenMeter} from '@deepseek-ai/dsh-token-meter'
import {SessionProjectionRegistry} from '@deepseek-ai/dsh-session-projection'
import * as plugin from '../lib/index.js'
const make=()=>Session.create(SessionId(crypto.randomUUID()))
function response(s,content=[{type:'reasoning',text:'private'},{type:'text',text:'answer'}]){
 return s.append('assistant/message',{turn:1,step:1,stream:[],message:createAssistantMessage({content,source:{provider:'deepseek-account',model:'test',replayState:{private:true}}})},{surfaceOp:'append'})
}
function fakeMeter(s){return {measure:()=>({logRevision:s.seq,baseline:{kind:'usage',tokens:60000,usage:{inputTokens:50000,outputTokens:10000,reasoningTokens:9000}},surfaceDeltaTokens:0,totalTokens:60000,surfaceTokens:133,nodes:[{seq:1,tokens:133,heuristicTokens:10}]}),contextPressure:()=>({pressureTokens:50000,contextWindow:100000})}}
test('meter replaces billed-output anchor without changing clean route prices or billing',()=>{
 const s=make();const meter=fakeMeter(s);const original=meter.measure;const undo=interceptMeter(meter,x=>x===s)
 const m=meter.measure(s);assert.equal(m.baseline.kind,'estimated');assert.equal(m.surfaceDeltaTokens,0);assert.equal(m.totalTokens,133)
 assert.equal(m.nodes[0].tokens,133);assert.equal(m.nodes[0].heuristicTokens,10)
 assert.equal(meter.contextPressure(s).pressureTokens,50000)
 assert.equal(meter.measure(make()).baseline.kind,'usage');undo();assert.equal(meter.measure,original)
 assert.equal(meter.measure(s).baseline.usage.reasoningTokens,9000)
})
test('tool schema cost is added once; route image/file node prices are preserved',()=>{
 const s=make();const meter=fakeMeter(s);meter.measure=()=>({logRevision:s.seq,baseline:{kind:'usage',tokens:9999},surfaceDeltaTokens:0,totalTokens:9999,surfaceTokens:7777,nodes:[{seq:1,tokens:7777,heuristicTokens:55}]})
 const undo=interceptMeter(meter,()=>true);const header={config:{provider:'deepseek-account',model:'x'},tools:[{name:'test',description:'test',parameters:{type:'object'}}]}
 const m=meter.measure(s,header);assert.equal(m.nodes[0].tokens,7777);assert.equal(m.nodes[0].heuristicTokens,55);assert.ok(m.totalTokens>7777);assert.equal(m.totalTokens,m.baseline.tokens);undo()
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
test('subagents are untouched and mid-stream unload restores normal subsequent settlement',()=>{
 const s=make();const child=Session.create(s.id,[],{...s.header,origin:'subagent'});const undo=interceptAppend(child);response(child);assert.equal(child.deriveMessages()[0].content.length,2);undo()
 const dispose=interceptAppend(s);dispose();response(s);assert.equal(s.deriveMessages()[0].content.length,2)
})
