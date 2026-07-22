# CLAUDE.md — soppelrom-3d

Analyserer Polycam 3D-skann av søppelrom: finner/måler rommet og de eksisterende
søppelkassene, og foreslår hvor **nye** kasser kan stå (miks av typer, inntil vegg, av
skyve-stien). Norsk prosjekt — svar og kommentarer på norsk.

## Kjøring

Python **3.12**-venv (Open3D støtter ikke 3.14). Kjør moduler fra repo-roten:

```
.venv\Scripts\python.exe -m src.<modul>     # README-form; `python -m src.<modul>` funker også her
```

Sentrale inngangspunkter:
- `src.dashboard` — hoved-GUI (liste over skann + resultatbilder + knapper).
- `src.place3d` — interaktiv 3D-visning av plassering + skyve-sti.
- `src.prepare_scan --pending` — bygg punktsky/mesh + auto-forslag for uforberedte skann.
- `src.annotate3d` — annoteringsverktøy (bruker godkjenner bokser her).
- Trening: `src.train_bins` (YOLO), `src.train_verifier` (PointNet++), `src.train_doors`,
  `src.train_place_prior`. Alle tar `--val-frac 0` for endelig modell på alle scener.

Ingen formell testsuite. Verifiser endringer ved å kjøre `pipeline.compute_scene(stem, bin_type)`
og rendre resultatet (`render.placements_over_scene` / `freespace_over_scene` / `annotated_topdown`)
og se på bildet.

## Data og stier — VIKTIG

- **All sti-logikk går via `src/paths.py`.** Ikke hardkod stier. Store/regenererbare data
  (`raw/ cache/ previews/ yolo_dataset/`) ligger **lokalt utenfor OneDrive** under
  `%LOCALAPPDATA%\soppelrom-3d` (overstyr med env `SOPPELROM_DATA_DIR`) — de skal ikke synkes
  eller sjekkes inn.
- **I git/GitHub:** `src/`, `models/*.pt`, `outputs/annotations/`, `outputs/entrances/`.
  Vektene (`models/*.pt`) SKAL pushes.
- I dette workspacet ligger venv i parent-mappa (`..\.venv`), delt med andre prosjekt.

## Ufravikelige regler

- **Forslag skal ALDRI auto-godkjennes.** Bare brukeren godkjenner (i dashbordet/annoteringsverktøyet).
  `prepare_scan` setter alltid status «foreslått», aldri «godkjent» — verifiseringsnettet kan bare
  *droppe* forslag, ikke godkjenne. Dette beskytter treningsdataen.
- **Aldri overskriv `outputs/annotations/`** fra pipelinen. Brukeren skal aldri måtte annotere på nytt.
- **`torch.load(..., weights_only=False)`** for våre egne sjekkpunkter (de inneholder numpy
  mean/std; PyTorch 2.6 defaulter til `weights_only=True` og kræsjer ellers). Lagre mean/std som
  `.tolist()`.

## Pipeline (dataflyt)

`pipeline.compute_scene(stem, bin_type) -> Scene` er den delte inngangen (brukt av både
dashboard-previews og `place3d`):

1. `loader.load_point_cloud` (cache-.ply hvis den finnes, ellers rekonstruer).
2. `backbone.analyze` — gravitasjonsjustering, gulv = laveste horisontale plan, **fotavtrykk fra
   høydebånd (0.25 m) rundt lokal gulvhøyde**, beholder alle store gulvbiter (ikke bare største).
3. `freespace.compute_free_space` — ledig vs opptatt gulv. **Hinderhøyde måles fra LOKAL bakke per
   rute (+12 cm)**, ikke ett globalt plan, så skrånende/ujevnt gulv ikke blir feil-rødt.
4. `doors` — automatiske innganger (lært); manuelt klikk (`set_entrance`) overstyrer.
   `is_enclosed` hopper over rom skannet med lukket dør.
5. `placement.find_placements` → `pack_placements` — se under.

## Plassering (`placement.py`) — aktivt område

`pack_placements` fyller ledig gulv med en miks av kassetyper, **én kasse om gangen**:
- **Skyve-sti** (`route_corridor`, Dijkstra) er en nesten rett rute fra inngang rundt hver kasse og
  er hellig — ingen ny kasse settes på den. **En kasses eget fotavtrykk teller som sti (blått)**,
  så plassering blokkerer aldri en annen kasses vei; stien beregnes på nytt etter HVER plassering.
- Kasser hugger rommets ytterkant/vegg, snappes til rom-aksen (står parallelt), og klynges tett
  mot vegg + eksisterende kasser. Rangering = geometriske regler (vegg/nær/størrelse) + lært prior.
- `place_prior` (`train_place_prior`) er en liten MLP trent på dine annoterte kasser (posisjon-
  features). Den er en LETT rangerings-vekt; geometrien gjør mesteparten. Kan bare posisjon, ikke
  retning (retning løses geometrisk).

Rutenett-konvensjon: `[Z rader, X kolonner]`, celle 0.05 m. `BIN_TYPES` = `(length, height, width)`.

## Git

Remote `github.com/davgei/soppelrom-3d.git`, branch `main` (brukerens arbeidsflyt pusher direkte
til main). Commit/push kun når brukeren ber om det. Avslutt commit-meldinger med
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
