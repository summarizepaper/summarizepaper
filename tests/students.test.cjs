const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function setup(fetch) {
  const handlers = {}, windowHandlers = {}, timers = new Map(); let tick=0;
  const status = {textContent:'Automatic saving', classList:{toggle(name,value){this[name]=value;}}};
  const button = {disabled:false};
  const form = {action:{value:'remove'},getAttribute(name){return name==='action'?'/private/lead/':null;},dataset:{unsaved:'Unsaved',saving:'Saving',saved:'Saved',error:'Failed',conflict:'Conflict'},
    elements:{status:{value:'explore'},notes:{value:'Initial'},remind_on:{value:''},revision:{value:0}},
    querySelector(selector){return selector==='[data-student-save-status]'?status:button;},
    reportValidity(){return true;},addEventListener(name,fn){handlers[name]=fn;}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/students.js','utf8'),{
    document:{readyState:'complete',querySelectorAll(selector){return selector==='[data-student-shortlist]'?[form]:[];}},
    window:{addEventListener(name,fn){windowHandlers[name]=fn;}},
    setTimeout(fn){timers.set(++tick,fn);return tick;},clearTimeout(id){timers.delete(id);},
    FormData:class{constructor(f){this.values=Object.fromEntries(Object.entries(f.elements).map(([k,v])=>[k,v.value]));}}, fetch});
  async function flush(){const callbacks=[...timers.values()];timers.clear();for(const fn of callbacks) await fn();}
  return {form,status,button,handlers,windowHandlers,flush};
}
const success = revision=>({ok:true,status:200,json:async()=>({revision})});
test('status and reminder automatically save with CSRF form data and revision',async()=>{
  let call;const s=setup(async(url,options)=>{call={url,options};return success(1);});
  s.form.elements.status.value='contacted';s.form.elements.remind_on.value='2027-01-10';
  s.handlers.change({type:'change'});assert.equal(s.status.textContent,'Unsaved');await s.flush();
  assert.equal(s.status.textContent,'Saved');assert.equal(s.form.elements.revision.value,1);
  assert.equal(call.url,'/private/lead/');assert.equal(call.options.headers.Accept,'application/json');assert.equal(call.options.credentials,'same-origin');
  assert.equal(call.options.body.values.remind_on,'2027-01-10');
});
test('edits during an in-flight save are queued with the acknowledged revision',async()=>{
  let resolve;const calls=[];const s=setup(async(url,options)=>{calls.push(options.body.values);if(calls.length===1)return new Promise(r=>resolve=r);return success(2);});
  s.form.elements.notes.value='First';s.handlers.input({type:'input'});const first=s.flush();
  s.form.elements.notes.value='Second';s.handlers.input({type:'input'});await s.flush();
  resolve(success(1));await first;await s.flush();
  assert.equal(calls.length,2);assert.equal(calls[1].revision,1);assert.equal(calls[1].notes,'Second');assert.equal(s.status.textContent,'Saved');
});
test('failed save preserves edits, warns before leaving and can retry',async()=>{
  let failed=true;const s=setup(async()=>{if(failed)throw Error();return success(1);});
  s.form.elements.notes.value='Keep me';s.handlers.input({type:'input'});await s.flush();assert.equal(s.status.textContent,'Failed');
  let prevented=false;s.windowHandlers.beforeunload({preventDefault(){prevented=true;}});assert.equal(prevented,true);
  failed=false;let submitted=false;s.handlers.submit({preventDefault(){submitted=true;},submitter:{value:''}});
  await new Promise(r=>setImmediate(r));assert.equal(submitted,true);assert.equal(s.status.textContent,'Saved');
});
test('conflict stops automatic retries without losing notes',async()=>{
  let calls=0;const s=setup(async()=>{calls++;return {status:409};});s.form.elements.notes.value='My notes';s.handlers.change({type:'change'});await s.flush();
  s.handlers.submit({preventDefault(){},submitter:{value:''}});await s.flush();assert.equal(calls,1);assert.equal(s.status.textContent,'Conflict');assert.equal(s.form.elements.notes.value,'My notes');
});
test('remove cancels pending saves; reverting a change sends nothing',async()=>{
  let calls=0;const s=setup(async()=>{calls++;return success(1);});s.form.elements.notes.value='New';s.handlers.input({type:'input'});
  s.form.elements.notes.value='Initial';s.handlers.input({type:'input'});await s.flush();assert.equal(calls,0);
  s.form.elements.notes.value='New';s.handlers.input({type:'input'});s.handlers.submit({submitter:{value:'remove'}});await s.flush();assert.equal(calls,0);
});
test('a reading question replaces the prompt before the signature and preserves edits',()=>{
  let click;const draft={value:'Subject\n\nMy edited introduction. [Reading prompt]\n\nThank you\n\nSignature',dataset:{questionPlaceholder:'[Reading prompt]',questionClosing:'Thank you'},closest(){return details;},focus(){this.focused=true;}};
  const details={open:false}, message={dataset:{added:'Added'}}, card={querySelector(selector){return selector==='[data-student-draft]'?draft:message;}};
  const button={dataset:{studentQuestion:'Specific question?'},closest(){return card;},addEventListener(name,fn){click=fn;}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/students.js','utf8'),{document:{readyState:'complete',querySelectorAll(selector){return selector==='[data-student-question]'?[button]:[];}}});
  click();click();assert.equal(draft.value,'Subject\n\nMy edited introduction. Specific question?\n\nThank you\n\nSignature');assert.equal(details.open,true);assert.equal(message.textContent,'Added');assert.equal(draft.focused,true);
});
