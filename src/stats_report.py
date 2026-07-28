"""Build a self-contained, interactive statistics page for all analysed scans.

Reads every previews/<stem>/stats.json plus the annotated bin types, computes per-scan and aggregate
numbers (including a fun "time to empty every bin" estimate), and writes ONE offline HTML file with
KPI tiles, charts (bin types, emptying-time, room-size, spare capacity, a top-list) and a sortable
room table. No external libraries — inline CSS/SVG/JS, so it opens in any browser and can be shown
straight off a laptop. Deliberately avoids importing pipeline (no open3d) so it stays fast.

    .venv\\Scripts\\python.exe -m src.stats_report          # writes + prints the path
"""
from __future__ import annotations

import json
import re
import webbrowser
from pathlib import Path

from .annotations import load_annotations
from .paths import ANNOTATION_DIR, CACHE_ROOT, PREVIEW_ROOT

# Rough seconds to empty one bin of each type (wheel out, hook/lift, tip, return). Adjustable —
# these drive the "time to empty every bin" statistics, clearly labelled as estimates on the page.
EMPTY_SECONDS: dict[str, int] = {
    "2-hjuls dunk": 20,
    "4-hjuls container": 40,
    "molok": 150,
    "annet": 30,
}

# Walking model: the crew walks in once to fetch the nearest bin, wheels each bin FULL to the truck
# and — after emptying — wheels it EMPTY back to its spot, then walks back out once. So a room with
# n wheeled bins has (2*n + 2) one-way legs (a 5-bin room = 6 round trips, not 10). Big bins roll
# faster than small ones; a full bin is slower than an empty one; a molok is emptied in place and
# never wheeled. Speeds match the 3D animation in place3d. All adjustable.
WHEEL_SPEED_FULL: dict[str, float] = {
    "4-hjuls container": 0.63,  # m/s wheeling a FULL bin to the truck
    "2-hjuls dunk": 0.52,
    "annet": 0.52,
}
WHEEL_SPEED_EMPTY: dict[str, float] = {
    "4-hjuls container": 0.92,  # m/s wheeling the EMPTIED bin back
    "2-hjuls dunk": 0.80,
    "annet": 0.80,
}
WALK_SPEED = 1.05        # m/s empty-handed (fetch the first bin + final walk back)
WALK_ONE_WAY_M = 12.0    # assumed one-way distance between the truck and the bins


def _bin_types(stem: str) -> list[str]:
    annotated = ANNOTATION_DIR / f"{stem}.json"
    proposals = CACHE_ROOT / stem / "proposals.json"
    path = annotated if annotated.exists() else (proposals if proposals.exists() else None)
    if path is None:
        return []
    try:
        _, boxes = load_annotations(path)
    except Exception:  # noqa: BLE001 - a malformed file should not kill the whole report
        return []
    return [box.bin_type for box in boxes]


def _postal(address: str | None) -> str:
    match = re.search(r"(\d{4})\s+\D", address or "")
    return match.group(1) if match else ""


def gather() -> list[dict]:
    records: list[dict] = []
    for stats_path in sorted(PREVIEW_ROOT.glob("*/stats.json")):
        stem = stats_path.parent.name
        try:
            st = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        types = _bin_types(stem)
        by_type: dict[str, int] = {}
        for t in types:
            by_type[t] = by_type.get(t, 0) + 1
        empty = sum(EMPTY_SECONDS.get(t, EMPTY_SECONDS["annet"]) for t in types)
        wheeled = [t for t in types if t in WHEEL_SPEED_FULL]
        if wheeled:
            walk = 2 * (WALK_ONE_WAY_M / WALK_SPEED)          # fetch first bin + final return (empty)
            for t in wheeled:                                 # full bin out + emptied bin back
                walk += WALK_ONE_WAY_M / WHEEL_SPEED_FULL[t] + WALK_ONE_WAY_M / WHEEL_SPEED_EMPTY[t]
        else:
            walk = 0.0
        walk = round(walk)
        records.append({
            "stem": stem,
            "address": st.get("address") or stem,
            "postal": _postal(st.get("address")),
            "indoor": bool(st.get("indoor")),
            "closed": bool(st.get("closed_room")),
            "annotated": (ANNOTATION_DIR / f"{stem}.json").exists(),
            "area": round(st.get("area_m2") or 0.0, 1),
            "free": round(st.get("free_area_m2") or 0.0, 1),
            "height": round(st.get("room_height_m") or 0.0, 2),
            "bins": len(types),
            "byType": by_type,
            "emptySec": empty,
            "walkSec": walk,
            "serviceSec": empty + walk,
            "candidates": int(st.get("n_candidates") or 0),
            "entrances": int(st.get("n_entrances") or 0),
        })
    return records


def build(out_path: Path | None = None) -> Path:
    records = gather()
    if out_path is None:
        out_path = PREVIEW_ROOT / "statistikk.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = (_TEMPLATE
            .replace("__DATA__", json.dumps(records, ensure_ascii=False))
            .replace("__EMPTY__", json.dumps(EMPTY_SECONDS, ensure_ascii=False)))
    out_path.write_text(html, encoding="utf-8")
    return out_path


_TEMPLATE = r"""<!doctype html>
<html lang="no">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Søppelrom – statistikk</title>
<style>
  :root{
    color-scheme: light;
    --surface:#fcfcfb; --page:#f3f3ef; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --good:#0ca30c;
  }
  @media (prefers-color-scheme: dark){:root{
    color-scheme: dark;
    --surface:#1a1a19; --page:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --good:#0ca30c;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--page);color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.4}
  .wrap{max-width:1200px;margin:0 auto;padding:28px 20px 60px}
  header h1{margin:0 0 4px;font-size:26px;font-weight:650}
  header p{margin:0;color:var(--ink2);font-size:14px}
  .filters{display:flex;gap:8px;margin:18px 0 22px;flex-wrap:wrap}
  .filters button{font:inherit;font-size:13px;padding:7px 14px;border-radius:999px;cursor:pointer;
    border:1px solid var(--border);background:var(--surface);color:var(--ink2)}
  .filters button.on{background:var(--s1);color:#fff;border-color:transparent;font-weight:600}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px}
  .kpi{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 16px 14px}
  .kpi .v{font-size:27px;font-weight:680;letter-spacing:-.5px}
  .kpi .l{font-size:12px;color:var(--ink2);margin-top:3px}
  .kpi .s{font-size:11px;color:var(--muted);margin-top:2px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
  @media (max-width:820px){.grid{grid-template-columns:1fr}}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 18px}
  .card h2{margin:0 0 2px;font-size:15px;font-weight:620}
  .card .sub{margin:0 0 12px;font-size:12px;color:var(--muted)}
  .legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px;font-size:12px;color:var(--ink2)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .legend i{width:11px;height:11px;border-radius:3px;display:inline-block}
  svg{display:block;width:100%;overflow:visible}
  .bar-row{font-size:12px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
  th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
  th{cursor:pointer;color:var(--ink2);font-weight:600;user-select:none;position:sticky;top:0;background:var(--surface)}
  th:hover{color:var(--ink)}
  tbody tr:hover{background:color-mix(in srgb,var(--s1) 8%,transparent)}
  .pill{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--border);color:var(--ink2)}
  .search{font:inherit;font-size:13px;padding:7px 12px;border-radius:9px;border:1px solid var(--border);
    background:var(--page);color:var(--ink);width:240px;max-width:100%}
  .tablecard{margin-top:16px;max-height:560px;overflow:auto}
  .note{font-size:11px;color:var(--muted);margin-top:10px}
  #tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);font-size:12px;
    padding:6px 9px;border-radius:7px;opacity:0;transition:opacity .08s;z-index:9;white-space:nowrap}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Søppelrom – statistikk</h1>
    <p id="subtitle"></p>
  </header>

  <div class="filters" id="filters">
    <button data-f="all" class="on">Alle</button>
    <button data-f="indoor">Innendørs</button>
    <button data-f="outdoor">Utendørs</button>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="grid">
    <div class="card"><h2>Kasser etter type</h2><p class="sub">Fordeling av alle registrerte kasser</p>
      <div id="donut"></div><div class="legend" id="donut-legend"></div></div>
    <div class="card"><h2>Tømmetid per kassetype</h2><p class="sub">Anslått samlet tid, fordelt på type</p>
      <div id="timebars"></div></div>
    <div class="card"><h2>Romstørrelse</h2><p class="sub">Antall rom per størrelsesintervall</p>
      <div id="sizehist"></div></div>
    <div class="card"><h2>Plass til nye kasser</h2><p class="sub">Antall rom etter hvor mange nye kasser som får plass</p>
      <div id="capbars"></div></div>
  </div>

  <div class="grid" style="grid-template-columns:1fr">
    <div class="card"><h2>Mest ledig gulv</h2><p class="sub">Topp 12 rom etter ledig gulvareal</p>
      <div id="leaderboard"></div></div>
  </div>

  <div class="card tablecard">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:12px;flex-wrap:wrap">
      <h2 style="margin:0">Alle rom</h2>
      <input class="search" id="search" placeholder="Søk adresse …">
    </div>
    <table><thead><tr id="thead"></tr></thead><tbody id="tbody"></tbody></table>
  </div>

  <p class="note" id="assumptions"></p>
</div>
<div id="tip"></div>

<script>
const SCANS = __DATA__;
const EMPTY = __EMPTY__;
const TYPES = ["4-hjuls container","2-hjuls dunk","molok","annet"];
const SVG="http://www.w3.org/2000/svg";
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const colorFor = t => cssv(["--s1","--s2","--s3","--s4"][Math.max(0,TYPES.indexOf(t))] || "--s4");
let filter="all";

function fmtTime(s){s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;
  if(h>0)return h+" t "+m+" min";if(m>0)return m+" min "+(sec?sec+" s":"");return sec+" s";}
const fmt = (n,d=0)=>n.toLocaleString("no-NO",{maximumFractionDigits:d});

function filtered(){return SCANS.filter(s=>filter==="all"||(filter==="indoor")===s.indoor);}

function aggregate(rows){
  const a={rooms:rows.length,bins:0,byType:{},emptyTotal:0,walk:0,service:0,area:0,free:0,fitOne:0,indoor:0,outdoor:0};
  for(const s of rows){
    a.bins+=s.bins;a.emptyTotal+=s.emptySec;a.walk+=s.walkSec;a.service+=s.serviceSec;
    a.area+=s.area;a.free+=s.free;
    if(s.candidates>=1)a.fitOne++; s.indoor?a.indoor++:a.outdoor++;
    for(const t in s.byType)a.byType[t]=(a.byType[t]||0)+s.byType[t];
  }
  return a;
}

const tip=document.getElementById("tip");
function showTip(e,html){tip.innerHTML=html;tip.style.opacity=1;moveTip(e);}
function moveTip(e){tip.style.left=(e.clientX+14)+"px";tip.style.top=(e.clientY+14)+"px";}
function hideTip(){tip.style.opacity=0;}
function hoverable(el,html){el.addEventListener("mouseenter",e=>showTip(e,html));
  el.addEventListener("mousemove",moveTip);el.addEventListener("mouseleave",hideTip);}

function el(tag,attrs){const n=document.createElementNS(SVG,tag);for(const k in attrs)n.setAttribute(k,attrs[k]);return n;}
function clear(id){const n=document.getElementById(id);n.innerHTML="";return n;}

function renderKPIs(a){
  const freePct=a.area?Math.round(100*a.free/a.area):0;
  const tiles=[
    ["Rom analysert",fmt(a.rooms),a.indoor+" inne · "+a.outdoor+" ute"],
    ["Kasser totalt",fmt(a.bins),(a.rooms?(a.bins/a.rooms).toFixed(1):0)+" per rom"],
    ["Total tid: tøm + gå",fmtTime(a.service),"tømming "+fmtTime(a.emptyTotal)+" · gåing "+fmtTime(a.walk)],
    ["Snitt per rom",fmtTime(a.rooms?a.service/a.rooms:0),"tømming + gåing"],
    ["Gulvareal",fmt(a.area)+" m²","kartlagt totalt"],
    ["Ledig gulv",fmt(a.free)+" m²",freePct+" % av arealet"],
    ["Plass til ny kasse",a.fitOne+" rom",(a.rooms?Math.round(100*a.fitOne/a.rooms):0)+" % av rommene"],
  ];
  const box=clear("kpis");
  for(const[l,v,s]of tiles){const d=document.createElement("div");d.className="kpi";
    d.innerHTML=`<div class="v">${v}</div><div class="l">${l}</div><div class="s">${s}</div>`;box.appendChild(d);}
}

function renderDonut(a){
  const box=clear("donut");const W=320,H=210,cx=W/2,cy=H/2,r=82,ir=48;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:210});box.appendChild(svg);
  const items=TYPES.filter(t=>a.byType[t]).map(t=>({t,n:a.byType[t]}));
  const total=items.reduce((s,i)=>s+i.n,0)||1;let ang=-Math.PI/2;
  for(const it of items){
    const frac=it.n/total,a1=ang,a2=ang+frac*2*Math.PI;ang=a2;
    const p=(rr,an)=>[cx+rr*Math.cos(an),cy+rr*Math.sin(an)];
    const large=frac>0.5?1:0;const[x1,y1]=p(r,a1),[x2,y2]=p(r,a2),[x3,y3]=p(ir,a2),[x4,y4]=p(ir,a1);
    const path=el("path",{d:`M${x1},${y1} A${r},${r} 0 ${large},1 ${x2},${y2} L${x3},${y3} A${ir},${ir} 0 ${large},0 ${x4},${y4} Z`,
      fill:colorFor(it.t),stroke:cssv("--surface"),"stroke-width":2});
    hoverable(path,`<b>${it.t}</b><br>${it.n} kasser · ${Math.round(100*frac)}%`);svg.appendChild(path);
  }
  const c=el("text",{x:cx,y:cy-4,"text-anchor":"middle","font-size":26,"font-weight":680,fill:cssv("--ink")});
  c.textContent=total;svg.appendChild(c);
  const cl=el("text",{x:cx,y:cy+15,"text-anchor":"middle","font-size":12,fill:cssv("--ink2")});cl.textContent="kasser";svg.appendChild(cl);
  const leg=clear("donut-legend");
  for(const it of items){const s=document.createElement("span");
    s.innerHTML=`<i style="background:${colorFor(it.t)}"></i>${it.t} · <b>${it.n}</b>`;leg.appendChild(s);}
}

function hbars(id,rows,unit,fmtVal){
  const box=clear(id);const W=440,rowH=30,pad=150,H=rows.length*rowH+8;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H});box.appendChild(svg);
  const max=Math.max(1,...rows.map(r=>r.v));
  rows.forEach((r,i)=>{
    const y=i*rowH+6,bw=(W-pad-70)*r.v/max;
    const lbl=el("text",{x:pad-8,y:y+13,"text-anchor":"end","font-size":12,fill:cssv("--ink2")});
    lbl.textContent=r.label;svg.appendChild(lbl);
    const bar=el("rect",{x:pad,y:y+3,width:Math.max(bw,2),height:16,rx:4,fill:r.color||cssv("--s1")});
    hoverable(bar,`<b>${r.label}</b><br>${fmtVal?fmtVal(r.v):fmt(r.v)+" "+unit}`);svg.appendChild(bar);
    const val=el("text",{x:pad+Math.max(bw,2)+8,y:y+15,"font-size":12,fill:cssv("--ink"),"font-weight":600});
    val.textContent=fmtVal?fmtVal(r.v):fmt(r.v);svg.appendChild(val);
  });
}

function renderTimeBars(a){
  const rows=TYPES.filter(t=>a.byType[t]).map(t=>({label:t,v:a.byType[t]*EMPTY[t],color:colorFor(t)}))
    .sort((x,y)=>y.v-x.v);
  hbars("timebars",rows,"",v=>fmtTime(v));
}

function renderSizeHist(rows){
  const edges=[0,25,50,100,150,200,1e9],labels=["<25","25–50","50–100","100–150","150–200","200+"];
  const counts=labels.map(()=>0);
  for(const s of rows){for(let i=0;i<labels.length;i++){if(s.area<edges[i+1]){counts[i]++;break;}}}
  vbars("sizehist",labels.map((l,i)=>({label:l,v:counts[i]})),"m²","rom");
}

function renderCapacity(rows){
  const labels=["0","1","2","3","4","5","6+"],counts=labels.map(()=>0);
  for(const s of rows){const c=Math.min(s.candidates,6);counts[c]++;}
  vbars("capbars",labels.map((l,i)=>({label:l,v:counts[i]})),"nye","rom");
}

function vbars(id,rows,xunit,yunit){
  const box=clear(id);const W=440,H=210,padB=34,padL=8,padT=10;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H});box.appendChild(svg);
  const max=Math.max(1,...rows.map(r=>r.v)),n=rows.length,gap=12,bw=(W-padL*2-gap*(n-1))/n;
  rows.forEach((r,i)=>{
    const x=padL+i*(bw+gap),h=(H-padB-padT)*r.v/max,y=H-padB-h;
    const bar=el("rect",{x,y,width:bw,height:Math.max(h,1),rx:4,fill:cssv("--s1")});
    hoverable(bar,`<b>${r.label} ${xunit}</b><br>${r.v} ${yunit}`);svg.appendChild(bar);
    const v=el("text",{x:x+bw/2,y:y-5,"text-anchor":"middle","font-size":12,fill:cssv("--ink"),"font-weight":600});
    v.textContent=r.v;svg.appendChild(v);
    const l=el("text",{x:x+bw/2,y:H-12,"text-anchor":"middle","font-size":12,fill:cssv("--ink2")});
    l.textContent=r.label;svg.appendChild(l);
  });
}

function renderLeaderboard(rows){
  const top=[...rows].sort((a,b)=>b.free-a.free).slice(0,12)
    .map(s=>({label:s.address.length>34?s.address.slice(0,33)+"…":s.address,v:s.free,color:cssv("--s3")}));
  hbars("leaderboard",top,"m²",v=>fmt(v,1)+" m²");
}

let sortKey="free",sortDir=-1,searchStr="";
const COLS=[["address","Adresse"],["kind","Type"],["area","Areal m²"],["free","Ledig m²"],
  ["bins","Kasser"],["serviceSec","Servicetid"],["candidates","Nye plasser"]];
function renderTable(rows){
  const head=clear("thead");
  for(const[k,l]of COLS){const th=document.createElement("th");th.textContent=l;
    if(k===sortKey)th.textContent+=sortDir<0?" ▾":" ▴";
    th.onclick=()=>{if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=(k==="address"?1:-1);}renderTable(filtered());};
    head.appendChild(th);}
  let data=rows.filter(s=>s.address.toLowerCase().includes(searchStr));
  data.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(sortKey==="kind"){x=a.indoor;y=b.indoor;}
    if(typeof x==="string")return sortDir*x.localeCompare(y,"no");return sortDir*((x>y)-(x<y));});
  const body=clear("tbody");
  for(const s of data){const tr=document.createElement("tr");
    tr.innerHTML=`<td>${s.address}${s.annotated?'':' <span class="pill">forslag</span>'}</td>`+
      `<td><span class="pill">${s.indoor?"inne":"ute"}</span></td>`+
      `<td>${fmt(s.area,1)}</td><td>${fmt(s.free,1)}</td><td>${s.bins}</td>`+
      `<td>${fmtTime(s.serviceSec)}</td><td>${s.candidates}</td>`;
    body.appendChild(tr);}
}

function renderAll(){
  const rows=filtered(),a=aggregate(rows);
  document.getElementById("subtitle").textContent=
    `${SCANS.length} skann analysert · ${aggregate(SCANS).bins} kasser · Oslo kommune (REG)`;
  renderKPIs(a);renderDonut(a);renderTimeBars(a);renderSizeHist(rows);renderCapacity(rows);
  renderLeaderboard(rows);renderTable(rows);
  const asu=Object.entries(EMPTY).map(([t,s])=>`${t} ${s} s`).join(", ");
  document.getElementById("assumptions").textContent=
    "Servicetid = tømmetid + gåtid. Tømmetid er anslag per kassetype ("+asu+"). Gåtid: crewet går inn "+
    "for å hente nærmeste kasse, triller hver kasse FULL til bilen og TOM tilbake, og går ut igjen — "+
    "(2 × antall kasser + 2) enveis-strekk à ~12 m (5 kasser = 6 tur/retur, ikke 10). Fart: full 4-hjuls "+
    "0,63 / full 2-hjuls 0,52 m/s; tom 4-hjuls 0,92 / tom 2-hjuls 0,80 m/s; tomhendt 1,05 m/s. Molok "+
    "trilles ikke. Alt kan justeres i src/stats_report.py. «forslag» = rom som ennå ikke er annotert.";
}

document.getElementById("filters").addEventListener("click",e=>{
  const b=e.target.closest("button");if(!b)return;
  filter=b.dataset.f;[...e.currentTarget.children].forEach(x=>x.classList.toggle("on",x===b));
  renderAll();
});
document.getElementById("search").addEventListener("input",e=>{searchStr=e.target.value.toLowerCase();renderTable(filtered());});
renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    path = build()
    print(f"Skrev {path}")
    try:
        webbrowser.open(path.as_uri())
    except Exception:  # noqa: BLE001 - opening a browser is best-effort
        pass


if __name__ == "__main__":
    main()
