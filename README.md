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

## Verifisering av forslag (PointNet++)

YOLO→tilbakeprojisering foreslår for mye — vegger, gulvrot og tynne «slivers» slipper gjennom
størrelses-porten i `binfit`. `src.verify_bins` er et lite PointNet++-nett som ser på de faktiske
3D-punktene *inne i* hver forslagsboks og gir P(kasse). `prepare_scan` bruker det til å droppe
sikre ikke-kasser og til å ikke auto-godkjenne usikre forslag. Punktene beholdes i meter (Y = høyde
over gulvet), så absolutt størrelse og «står på gulvet» blir signaler nettet kan bruke.

Tren fra dine egne 3D-annoteringer (positive = annoterte kasser, negative = pipelinens egne
bomskudd + tilfeldige bokser). Modellen lagres som `models/verifier_latest.pt` og plukkes opp
automatisk av `prepare_scan` (fjern fila for å skru verifiseringen av).

```
.venv\Scripts\python.exe -m src.train_verifier            # holder av noen scener til validering
.venv\Scripts\python.exe -m src.train_verifier --val-frac 0   # tren på alle scener (endelig modell)
```

Nettet er lite (~105k parametre, ren PyTorch) og trener/kjører på GPU hvis CUDA-torch er installert
(se «Miljø»), ellers på CPU. Terskler for å droppe/sende-til-gjennomgang står øverst i `verify_bins.py`.

## Dør-deteksjon (automatisk)

`src.doors` finner innganger automatisk, lært fra dørene du har klikket (`outputs/entrances/`).
Dører er *åpninger* (fravær av vegg), så en punkt-modell passer ikke — i stedet samples kandidat-
punkter langs hele gulv-perimeteret, hver beskrevet med enkle features (hvor mye vegg her, om åpen
plass lekker utover, om skanneren gikk gjennom, kamera-trafikk), og en liten klassifikator avgjør
hvilke som er ekte dører. En «gikk-gjennom»-port kutter falske dører på vegger ingen var i nærheten av.

```
.venv\Scripts\python.exe -m src.train_doors --val-frac 0    # tren fra klikkede dører
```

Modellen lagres som `models/doors_latest.pt` og brukes automatisk av pipelinen (`analyze_and_render`).
Finnes ingen modell, faller den tilbake til den geometriske heuristikken. Manuelt klikk (`set_entrance`
/ annoteringsverktøyet) overstyrer fortsatt. Terskelen `KEEP_PROB` i `doors.py` styrer presisjon vs
å finne alle dører (nå satt lavt for å heller ta med en dør for mye enn å blokkere en ekte). Modellen
blir bedre jo flere dører du klikker og retrener på.

## Annotering (Fase 2)

Annoteringsverktøyet viser Poisson-meshen med forslags-bokser fra zero-shot-deteksjonen
(oransje = forslag, grønn = godkjent, blå = valgt). CAD-stil: «Tegn boks» = klikk to
hjørner på gulvet, trykk for dybde og dra opp for høyde. Valgt boks har håndtak —
dra gult hjørne (størrelse), blå topp (høyde), rosa kule (rotasjon), eller selve boksen
(flytt). Klikk på en boks velger den; ESC avbryter; Ctrl+Z angrer. Annoteringer lagres i
`outputs/annotations/`, og en bakgrunnsprosess holder inntil 5 skann ferdig forberedt.

Slipper å tegne når målet er kjent: **«Plasser boks» (P)** — klikk på gulvet der kassa står,
så settes en boks med kassetypens faste mål (2-/4-hjuls er alltid like store). Klikk videre for
flere; trykk knappen igjen, **P** eller **ESC** for å avslutte. **Ctrl+C / Ctrl+V** kopierer og
limer inn valgt boks (kopien havner ved siden av originalen), nyttig for rader med like kasser.
