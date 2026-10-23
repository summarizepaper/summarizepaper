const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness({fragmentOK = true, command = 'start_background_task'} = {}) {
  const events = {}, requests = [], sockets = [], tabs = [], panels = [];
  const el = () => ({dataset:{}, style:{}, hidden:false, textContent:'', children:[],
    classList:{add() {}}, addEventListener(t, fn) {this[t] = fn;},
    setAttribute() {}, replaceChildren(...nodes) {this.children = nodes; this.textContent = '';},
    appendChild(node) {this.children.push(node);}});
  const status = el(), retry = el(), progress = el(), page = el(), title = el();
  for (let i=0; i<4; i++) {tabs.push(Object.assign(el(), {dataset:{liveTab:String(i)}})); panels.push(el());}
  const scope = Object.assign(el(), {querySelector: () => el()});
  const box = Object.assign(el(), {dataset:{command, state:'/state', fragment:'/detail', canonical:'/paper/', paper:'id', lang:'fr', retry:'Retry'},
    querySelector(s) {return {'#live-status':status, '#live-retry':retry, progress, '.source-scope':scope, '#live-title':title}[s] ||
      (s.startsWith('#live-panel-') ? panels[Number(s.slice(-1))] : s.startsWith('#live-tab-') ? tabs[Number(s.slice(-1))] : null);},
    querySelectorAll: () => tabs});
  const removedScripts = [];
  const detail = {querySelectorAll: () => [{remove() {removedScripts.push(true);}}]};
  let inits = 0, replacedURL = '', selected = -1, fragmentFailure = !fragmentOK;
  const context = {console, location:{protocol:'http:', host:'local', hash:'', assign() {throw Error('unexpected navigation');}},
    document:{getElementById: id => id === 'synthesis-live' ? box : page, createElement:el, importNode: x=>x},
    window:{scrollY:12, scrollTo() {}, addEventListener() {}},
    history:{state:null, replaceState(_,__,url) {replacedURL = url;}},
    SPPaper:{init() {inits++;}}, switchTab(i) {selected=i;},
    DOMParser: class {parseFromString() {return {querySelector: () => detail};}},
    setTimeout(fn) {events.reconnect=fn; return 1;}, clearTimeout() {},
    setInterval(fn) {events.poll=fn; return 2;}, clearInterval() {},
    WebSocket: class {constructor() {sockets.push(this); this.sent=[];} send(data) {this.sent.push(JSON.parse(data));} close() {this.closed=true;}},
    async fetch(url) {requests.push(url); if(url==='/detail') return {ok:!fragmentFailure, text:async()=>'<div/>'}; return {ok:true,json:async()=>({status:'running'})};}
  };
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/paper-live.js','utf8'),context);
  function emit(state) {sockets.at(-1).onmessage({data:JSON.stringify({type:'synthesis_snapshot',state})});}
  return {box, status, progress, retry, page, panels, sockets, tabs, events, requests, emit, removedScripts,
    fixed() {fragmentFailure=false;}, get inits(){return inits;}, get url(){return replacedURL;}, get selected(){return selected;}};
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('live summaries remain next to progress and a manual tab selection is retained', () => {
  const h=harness(); h.sockets[0].onopen();
  h.emit({summary:'First result',progress:{progress:50,loading_message:'Writing'}});
  assert.equal(h.panels[0].textContent,'First result'); assert.equal(h.progress.value,50); assert.equal(h.selected,0);
  h.emit({lay_summary:'Second result'}); assert.equal(h.selected,0);
  h.tabs[0].click(); h.emit({lay_summary:'Accessible result',progress:{progress:70}});
  assert.equal(h.panels[2].textContent,'Accessible result'); assert.equal(h.selected,0);
});

test('reconnection resumes snapshots without replaying the authorized start', () => {
  const h=harness(); h.sockets[0].onopen();
  assert.equal(h.sockets[0].sent[0].command,'start_background_task');
  h.sockets[0].onclose(); h.events.reconnect(); h.sockets[1].onopen();
  assert.equal(h.sockets[1].sent.length,0);
});

test('completed synthesis installs the final controls once without a navigation or executable scripts', async () => {
  const h=harness(); h.emit({progress:{progress:100}}); await settle();
  assert.equal(h.inits,1); assert.equal(h.page.children.length,1); assert.equal(h.url,'/paper/');
  assert.equal(h.removedScripts.length,1); assert.equal(h.sockets[0].closed,true);
  h.emit({progress:{progress:100}}); await settle(); assert.equal(h.inits,1);
});

test('failed final fetch preserves partial results and the retry installs tools', async () => {
  const h=harness({fragmentOK:false}); h.emit({summary:'Keep this',progress:{progress:100}}); await settle();
  assert.equal(h.panels[0].textContent,'Keep this'); assert.equal(h.inits,0); assert.equal(h.retry.hidden,false);
  h.fixed(); h.retry.click(); await settle(); assert.equal(h.inits,1);
});
