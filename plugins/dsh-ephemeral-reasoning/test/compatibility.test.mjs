import assert from 'node:assert/strict'
import {test} from 'node:test'
import {Context} from '@deepseek-ai/cordis'
import {Session,SessionId} from '@deepseek-ai/dsh-session'
import {TokenMeter} from '@deepseek-ai/dsh-token-meter'
import {SessionProjectionRegistry} from '@deepseek-ai/dsh-session-projection'
import {assertVersions,assertSessionCompatible,assertMeterCompatible,SUPPORTED_DSH_VERSION} from '../lib/compatibility.js'
import * as plugin from '../lib/index.js'
const names=['agent','session','llm','token-meter'].map(n=>'@deepseek-ai/dsh-'+n)
const versions=Object.fromEntries(names.map(n=>[n,SUPPORTED_DSH_VERSION]))
test('exact package versions are required, including every prerelease component',()=>{
 assert.doesNotThrow(()=>assertVersions())
 assert.doesNotThrow(()=>assertVersions(versions))
 for(const name of names){
  for(const version of ['0.2.2-rc.2','0.2.2','0.2.1-rc.1']) assert.throws(()=>assertVersions({...versions,[name]:version}),new RegExp(name.replaceAll('.','\\.')+'@0\\.2\\.2-rc\\.1'))
  const missing={...versions};delete missing[name];assert.throws(()=>assertVersions(missing),/found missing/)
 }
})
test('actual incompatible or previously wrapped session/meter implementations are rejected',()=>{
 const session=Session.create(SessionId(crypto.randomUUID()));assertSessionCompatible(session)
 session.append=function(){throw Error('unexpected')};assert.throws(()=>assertSessionCompatible(session),/Session.append/)
 assert.throws(()=>assertMeterCompatible({measure(){}}),/TokenMeter.measure/)
})
test('startup validates all existing sessions before patching the real Cordis meter',async()=>{
 const ctx=new Context();const projections=ctx.plugin(SessionProjectionRegistry);await projections.await()
 const meter=ctx.plugin(TokenMeter);await meter.await()
 const session=Session.create(SessionId(crypto.randomUUID()));session.append=function(){throw Error('foreign wrapper')}
 const before=Object.getOwnPropertyDescriptor(ctx.tokenMeter,'measure')
 ctx.provide('agents',{list:()=>[{id:session.id,session,ctx}]})
 const fiber=ctx.plugin(plugin,{})
 await assert.rejects(fiber.await(),/Session.append/)
 assert.deepEqual(Object.getOwnPropertyDescriptor(ctx.tokenMeter,'measure'),before)
 assert.equal(Reflect.get(ctx.tokenMeter,Symbol.for('dsh-ephemeral-reasoning.owner')),undefined)
 await fiber.dispose();await meter.dispose();await projections.dispose()
})
