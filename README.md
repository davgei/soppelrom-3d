# soppelrom-3d

Analyse av 3D-skanninger (Polycam) av søppelrom: romtype (dedikert søppelrom?),
inne/ute, antall og størrelse på søppelkasser, romdimensjoner, og ledig plass til
en ny kasse. Resultatene vises som en roterbar 3D-scene i Rerun.

## Sett opp på ny PC

```
git clone https://github.com/davgei/soppelrom-3d.git
cd soppelrom-3d
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Repoet inneholder **kode + annoteringer + innganger + den trente modellen** (`models/bins_latest.pt`).
Rå-skannene (`data/raw/`, ~2,3 GB) er for store for GitHub — hent dem på den nye PC-en med
`python -m src.download_polycam --auto`, eller kopier `data/raw/`-mappa via minnepinne/OneDrive.
`outputs/cache/` og `outputs/previews/` regenereres automatisk (menyens «Generer alle»).

## Miljø

Prosjektet har sitt **eget** virtuelt miljø på **Python 3.12** (Open3D støtter ikke 3.14).
Dette er separat fra `trash-bin-detection`.

```
.venv\Scripts\python.exe        # prosjektets Python
```

Kjør moduler slik (fra prosjektroten):

```
.venv\Scripts\python.exe -m src.build_pointcloud --scan data\raw\<scan>.zip --save outputs\<scan>.ply
```

## Datastruktur

```
data/raw/     originale Polycam-eksporter (zip = rå keyframes; evt. .ply) — read-only
outputs/      genererte punktskyer, previews, resultater
src/          kildekode
configs/      konfigurasjon
```

## Punktsky (Fase 0/1)

`src.build_pointcloud` lager en farget, metrisk punktsky per skann:
bruker en eksportert `.ply` hvis den ligger ved siden av zip-en, ellers rekonstruerer
den fra rå RGB-D + kamera-poser.

Nyttige flagg: `--view` (åpne Rerun), `--render-dir <dir>` (ortho-previews),
`--voxel`, `--min-confidence`, `--max-depth`, `--convention {arkit,opencv}`.

## Deteksjon og annotering (Fase 2)

```
.venv\Scripts\python.exe -m src.detect_bins3d --scan data\raw\<scan>.zip --view   # 3D-deteksjon
.venv\Scripts\python.exe -m src.prepare_scan --pending                            # forbered alle skann
.venv\Scripts\python.exe -m src.annotate3d                                        # annoteringsverktøy
```

## Kontrollpanel (meny/GUI)

```
.venv\Scripts\python.exe -m src.dashboard
```

Én meny for alt: liste over alle skann (status: rå / klar / annotert), og for det valgte
skannet ser du resultatbildene — **Rom + mål**, **Ledig gulv**, **Plassering (ny kasse)** —
med kassetype-velger og en statistikk-linje (adresse, mål, inne/ute, antall kasser, ledig
areal, mulige nye plasser). Knapper: Generer bilder (denne / alle), Åpne i 3D, Annotér,
Sett inngang, Forbered. Pil venstre/høyre blar mellom skann. Ingen zip-navn i terminalen.

## Annotering (Fase 2)

Annoteringsverktøyet viser Poisson-meshen med forslags-bokser fra zero-shot-deteksjonen
(oransje = forslag, grønn = godkjent, blå = valgt). CAD-stil: «Tegn boks» = klikk to
hjørner på gulvet, trykk for dybde og dra opp for høyde. Valgt boks har håndtak —
dra gult hjørne (størrelse), blå topp (høyde), rosa kule (rotasjon), eller selve boksen
(flytt). Klikk på en boks velger den; ESC avbryter; Ctrl+Z angrer. Annoteringer lagres i
`outputs/annotations/`, og en bakgrunnsprosess holder inntil 5 skann ferdig forberedt.
