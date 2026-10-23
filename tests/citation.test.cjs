const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function harness({supported=true,clipboard=true}={}) {
 const events={},requests=[],writes=[];
 const el=()=>({hidden:false,disabled:false,value:'',events:{},addEventListener(t,fn){this.events[t]=fn;},focus(){this.focused=true;},select(){this.selected=true;}});
 const text=el(),status=el(),copy=el(),download=el(),retry=el(),close=el();
 const dialog=Object.assign(el(),{dataset:{loading:'Loading',failed:'Failed',copied:'Copied',copyFailed:'Manual'},open:false,
  showModal:supported?function(){this.open=true;}:undefined,close(){this.open=false;this.events.close();},
  querySelector:s=>({'[data-citation-text]':text,'[data-citation-status]':status,'[data-citation-copy]':copy,'[data-citation-download]':download,'[data-citation-retry]':retry,'[data-citation-close]':close}[s])});
 const context={document:{getElementById:()=>dialog,addEventListener(t,fn){events[t]=fn;}},navigator:clipboard?{clipboard:{writeText:async value=>writes.push(value)}}:{},
 fetch:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}))};
 vm.runInNewContext(fs.readFileSync('static/summarizer/js/citation.js','utf8'),context);
 const link=Object.assign(el(),{href:'/fr/bibtex/paper/',isConnected:true});
 return {events,requests,writes,dialog,text,status,copy,download,retry,close,link,
 click(options={}){const e={target:{closest:()=>link},preventDefault(){this.prevented=true;},button:0,...options};if(events.click)events.click(e);return e;},
 loaded(value='@misc{safe, title={<img src=x>}}'){requests.at(-1).resolve({ok:true,headers:{get:()=> 'text/plain; charset=utf-8'},text:async()=>value});}};
}
const settle=()=>new Promise(r=>setImmediate(r));
test('opens in place, renders BibTeX as plain text, copies it and restores focus on close',async()=>{
 const h=harness();assert.equal(h.click().prevented,true);assert.equal(h.dialog.open,true);assert.equal(h.copy.disabled,true);
 h.loaded();await settle();assert.match(h.text.value,/<img src=x>/);assert.equal(h.text.hidden,false);
 assert.equal(h.download.href,'/fr/bibtex/paper/?download=1');
 h.copy.events.click();await settle();assert.equal(h.writes[0],h.text.value);assert.equal(h.status.textContent,'Copied');
 h.close.events.click();assert.equal(h.link.focused,true);assert.equal(h.dialog.open,false);
});
test('failed requests allow retry and clipboard failures select the citation for manual copying',async()=>{
 const h=harness({clipboard:false});h.click();h.requests[0].reject(Error('offline'));await settle();
 assert.equal(h.status.textContent,'Failed');assert.equal(h.retry.hidden,false);assert.equal(h.copy.disabled,true);
 h.retry.events.click();h.loaded();await settle();h.copy.events.click();await settle();
 assert.equal(h.status.textContent,'Manual');assert.equal(h.text.selected,true);
});
test('closed or replaced requests cannot show stale content and HTML error pages are rejected',async()=>{
 const h=harness();h.click();h.close.events.click();h.click();
 h.requests[0].resolve({ok:true,headers:{get:()=> 'text/plain'},text:async()=> 'Stale'});await settle();assert.equal(h.text.value,'');
 h.requests[1].resolve({ok:true,headers:{get:()=> 'text/html'},text:async()=> '<html>Login</html>'});await settle();assert.equal(h.status.textContent,'Failed');
});
test('modified clicks and browsers without modal support keep the existing navigation fallback',()=>{
 const h=harness();assert.equal(h.click({ctrlKey:true}).prevented,undefined);assert.equal(h.requests.length,0);
 assert.equal(harness({supported:false}).click().prevented,undefined);
});
