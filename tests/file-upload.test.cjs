const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() {this.events={};this.hidden=true;this.textContent='';this.classes=new Set();this.attributes={};this.classList={add:x=>this.classes.add(x),remove:x=>this.classes.delete(x),toggle:(x,on)=>on?this.classes.add(x):this.classes.delete(x)};}
  addEventListener(type,fn) {(this.events[type] ||= []).push(fn);}
  dispatchEvent(event) {(this.events[event.type]||[]).forEach(fn=>fn(event));}
  setAttribute(key,value) {this.attributes[key]=value;}
  focus(){this.focused=true;}
}
const file=(name='references.bib',size=100,type='')=>({name,size,type});
const event=(type,files,target={})=>({type,target,preventDefault(){this.prevented=true;},dataTransfer:{types:['Files'],files,dropEffect:''}});
function harness(count=1, pdf=false) {
  const doc=new Element();doc.documentElement={lang:'fr'};
  const boxes=Array.from({length:count},()=>{
    const box=new Element(), input=new Element(), zone=new Element(), status=new Element(), error=new Element(), clear=new Element();
    input.accept='.bib,text/plain';input.form=new Element();input._files=[];
    Object.defineProperty(input,'files',{get(){return this._files;},set(value){if(this.rejectFiles)throw Error('unsupported');this._files=value;}});
    Object.defineProperty(input,'value',{set(value){if(value==='')this._files=[];}});
    box.dataset={maxBytes:'5242880',maxMb:'5',requiredExtension:'',selected:'Selected {name} ({size})',remove:'Remove {name}',errorCount:'One file',errorType:'Type {formats}',errorSize:'Max {max}',errorBrowser:'Use picker',kb:'Ko',mb:'Mo',bytes:'octets'};
    box.querySelector=s=>({'input[type="file"]':input,'[data-file-drop]':zone,'[data-file-status]':status,'[data-file-error]':error,'[data-file-clear]':clear}[s]);
    return Object.assign(box,{input,zone,status,error,clear});
  });
  if (pdf) for (const b of boxes) {b.input.accept='application/pdf,.pdf';b.dataset.requiredExtension='.pdf';b.dataset.maxBytes='20971520';b.dataset.maxMb='20';}
  doc.querySelectorAll=()=>boxes;
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/file-upload.js','utf8'),{document:doc,Intl,Event:class{constructor(type){this.type=type;}},setTimeout:fn=>fn()});
  doc.dispatchEvent({type:'DOMContentLoaded'});
  return {doc,boxes,drop(box,files){const e=event('drop',files);box.zone.dispatchEvent(e);return e;},choose(box,files){box.input._files=files;box.input.dispatchEvent({type:'change'});}};
}
test('drops attach a file only to the target field, show safe filename and size, and never submit',()=>{
  const h=harness(2), b=h.boxes[1], f=file('<img>.BIB',2048);let submitted=0;
  b.input.form.addEventListener('submit',()=>submitted++);
  assert.equal(h.drop(b,[f]).prevented,true);
  assert.equal(b.input.files[0],f);assert.equal(h.boxes[0].input.files.length,0);
  assert.equal(b.status.textContent,'Selected <img>.BIB (2 Ko)');assert.equal(b.clear.hidden,false);assert.equal(submitted,0);
  assert.equal(b.clear.attributes['aria-label'],'Remove <img>.BIB');
});
test('invalid drops preserve previous selection and display format, count and size errors',()=>{
  const h=harness(),b=h.boxes[0],f=file();h.drop(b,[f]);
  for(const [files,message] of [[[file('bad.exe')],'Type .bib,text/plain'],[[f,f],'One file'],[[file('big.bib',5242881)],'Max 5'],[[],'One file']]) {
    h.drop(b,files);assert.equal(b.input.files[0],f);assert.equal(b.error.textContent,message);assert.equal(b.error.hidden,false);
  }
  h.drop(b,[file('limit.bib',5242880)]);assert.equal(b.error.hidden,true);
});
test('native picker validates replacements, supports text/plain and keeps previous valid files',()=>{
  const h=harness(),b=h.boxes[0],f=file('references.txt',10,'text/plain');h.choose(b,[f]);
  h.choose(b,[file('bad.exe')]);assert.equal(b.input.files[0],f);assert.equal(b.error.hidden,false);
  h.choose(b,[file('new.bib')]);assert.equal(b.error.hidden,true);assert.equal(b.input.files[0].name,'new.bib');
});
test('PDF requires .pdf even with a misleading MIME, and accepts a PDF with no MIME',()=>{
  const h=harness(1,true),b=h.boxes[0];
  h.drop(b,[file('bad.exe',1,'application/pdf')]);assert.equal(b.input.files.length,0);
  h.drop(b,[file('DRAFT.PDF',20971520)]);assert.equal(b.input.files[0].name,'DRAFT.PDF');
  h.drop(b,[file('large.pdf',20971521)]);assert.equal(b.error.textContent,'Max 20');
});
test('clear and form reset remove stale status and errors and return focus to picker',()=>{
  const h=harness(),b=h.boxes[0];h.drop(b,[file()]);b.clear.dispatchEvent({type:'click'});
  assert.equal(b.input.files.length,0);assert.equal(b.status.hidden,true);assert.equal(b.clear.hidden,true);assert.equal(b.input.focused,true);
  h.drop(b,[file('bad.exe')]);b.input.form.dispatchEvent({type:'reset'});assert.equal(b.error.hidden,true);
});
test('file drags highlight nested targets, and outside file drops cannot navigate away; text drags are untouched',()=>{
  const h=harness(),b=h.boxes[0];b.zone.dispatchEvent(event('dragenter',[]));b.zone.dispatchEvent(event('dragenter',[]));
  b.zone.dispatchEvent(event('dragleave',[]));assert.equal(b.zone.classes.has('is-dragover'),true);
  b.zone.dispatchEvent(event('dragleave',[]));assert.equal(b.zone.classes.has('is-dragover'),false);
  const outside=event('dragover',[]);h.doc.dispatchEvent(outside);assert.equal(outside.prevented,true);assert.equal(outside.dataTransfer.dropEffect,'none');
  const drop=event('drop',[file()]);h.doc.dispatchEvent(drop);assert.equal(drop.prevented,true);
  const text=event('drop',[]);text.dataTransfer.types=['text/plain'];b.zone.dispatchEvent(text);h.doc.dispatchEvent(text);assert.equal(text.prevented,undefined);
});
test('unsupported FileList assignment explains the native picker fallback',()=>{
  const h=harness(),b=h.boxes[0];b.input.rejectFiles=true;h.drop(b,[file()]);assert.equal(b.error.textContent,'Use picker');assert.equal(b.input.files.length,0);
});
