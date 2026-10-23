const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness(auto = true) {
  const events={}, listeners={}, requests=[], waits=[], panels=[], removed=[];
  const button={disabled:false}, progress={hidden:true}, error={hidden:true};
  const questions={children:[], replaceChildren(){this.children=[];}, appendChild(li){this.children.push(li);}};
  const box={dataset:{auto:auto?'1':'0',start:'/start',result:'/result'},
    querySelector(s){return {'[data-publication-refresh]':button,'[data-publication-progress]':progress,'[data-publication-error]':error}[s];},
    addEventListener(t,fn){listeners[t]=fn;}, replaceChildren(...nodes){panels.push(nodes);}};
  const SP={post:(url,data)=>new Promise((resolve,reject)=>requests.push({url,data,resolve,reject})),wait:(...args)=>waits.push(args)};
  const math=[];
  const MathJax={typesetClear(){},typesetPromise(nodes){math.push(nodes);return Promise.resolve();}};
  const context={MathJax, document:{addEventListener:(t,fn)=>events[t]=fn,getElementById:id=>id==='publication-status'?box:questions,
    createElement:()=>({}),importNode:node=>node}, window:{MathJax}, SP,
    fetch:async()=>({ok:true,json:async()=>({html:'verified source HTML',questions:['Which version should be cited?','<img src=x>']})}),
    DOMParser:class {parseFromString(){return {body:{childNodes:[{textContent:'verified result'}]},querySelectorAll:()=>[{remove(){removed.push(true);}}]};}}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/publication-status.js','utf8'),context);
  events.DOMContentLoaded();
  return {button,progress,error,requests,waits,panels,questions,removed,math,
    click(){listeners.click({target:{closest:()=>button}});}};
}
const settle=()=>new Promise(resolve=>setImmediate(resolve));

test('old kits automatically check metadata with a spinner, prevent duplicates, and update questions without reloading',async()=>{
  const h=harness();
  assert.equal(h.requests.length,1); assert.equal(h.button.disabled,true); assert.equal(h.progress.hidden,false);
  h.click(); assert.equal(h.requests.length,1);
  h.requests[0].resolve({id:9}); await settle(); h.waits[0][1](); await settle();
  assert.equal(h.panels.length,1); assert.equal(h.removed.length,1);
  assert.equal(h.questions.children[1].textContent,'<img src=x>');
  assert.ok(h.math[0].includes(h.questions), 'retains LaTeX rendering when questions are replaced');
  assert.equal(h.button.disabled,false); assert.equal(h.progress.hidden,true); assert.equal(h.error.hidden,true);
});

test('recent checks do not repeat on page load, and failures retain the kit and allow retry',async()=>{
  const h=harness(false); assert.equal(h.requests.length,0);
  h.click(); h.requests[0].reject(Error('offline')); await settle();
  assert.equal(h.panels.length,0); assert.equal(h.button.disabled,false); assert.equal(h.error.hidden,false);
  h.click(); h.requests[1].resolve({id:2}); await settle(); h.waits[0][2]();
  assert.equal(h.panels.length,0); assert.equal(h.progress.hidden,true);
  assert.equal(h.button.disabled,false); assert.equal(h.error.hidden,false);
});
