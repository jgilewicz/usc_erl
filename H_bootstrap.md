# H-bootstrap: niepewność steruje długością rolloutu

Propozycja następnego mechanizmu SC-ERL. Motywacja i projekt; wyniki symulacji
offline na końcu.

## Motywacja (diagnoza z 2026-09-07)

- Ensemble krytyków jest poprawnym estymatorem epistemicznym błędu **krytyka**
  (σ rośnie monotonicznie z odległością od danych, patrz
  `scripts/probe_ensemble_uncertainty.py`).
- Błąd **surogatu** na osobnikach GA jest jednak zdominowany przez niedopasowanie
  rozkładu stanów: Q liczone na stanach z bufora vs zwrot na własnej trajektorii
  mutanta. Hopper: Spearman(surogat, zwrot) 0.11–0.46 na stanach z bufora,
  0.73–0.86 na własnej trajektorii; pula stanów populacji nie domyka luki.
- Tego błędu niepewność krytyka nie widzi z definicji, bo krytyk nie wie, dokąd
  mutant pójdzie. Stąd `gate_auc_u` ≈ 0.3–0.5. Sufit jest po stronie celu
  pomiaru, nie estymatora.
- Wniosek: niepewność powinna decydować **jak długo** ewaluować kandydata, a nie
  **czy** ewaluować go naprawdę. Wtedy każdy stan użyty w estymacie leży na
  własnej trajektorii kandydata, a σ mierzy dokładnie to, co zostaje: błąd
  krytyka w `s_H`.

## 1. Funkcja przystosowania z H-bootstrapu

Cel to niedyskontowany zwrot (ten sam, którego GA używa dziś), żeby estymaty
przy różnych H były na jednej skali i żeby przy `H = T` wychodził realny fitness.

```
po H krokach własnego rolloutu (nagrody r_0..r_{H-1}, stan s_H, flaga done_H):

F̂_i(H) = Σ_{t<H} r_t  +  (1 − done_H) · (T − H) · (1 − γ) · Q_i(s_H, π(s_H))   dla członka i

F̂(H)   = mean_i F̂_i(H)
σ(H)    = std_i F̂_i(H)  =  (1 − done_H) · (T − H) · (1 − γ) · std_i Q_i(s_H, π(s_H))
fitness = F̂(H) − β · σ(H)        (LCB jak dziś, β adaptacyjna jak dziś)
```

- `(1 − γ) · Q` to estymata średniej nagrody na krok (dokładna dla stacjonarnej
  nagrody), pomnożona przez liczbę pozostałych kroków.
- `H = T` → ogon znika, `F̂` = realny zwrot. `H = 0` → dzisiejszy surogat liczony
  na `s_0`.
- Wariant ablacyjny (`tail=disc`): `Σ γ^t r_t + γ^H · Q_i(s_H, ·)`. Semantycznie
  czystszy dla krytyka, ale estymuje zwrot **dyskontowany**, a GA optymalizuje
  niedyskontowany. Na Swimmerze `spearman(zwrot dysk., zwrot niedysk.)` = 0.54
  (66k) i 0.73 (110k) — to twardy sufit tego wariantu, niezależny od H.
- **Ogon musi być kalibrowany** (`tail=cal`). Surowe `(1 − γ) · Q` opiera się na
  absolutnej skali krytyka, a ta bywa rozjechana o rząd wielkości (Swimmer: Q ≈ 14
  przy realnym zwrocie ≈ 15 na 1000 kroków, czyli ok. 10× za dużo). Wtedy ogon
  dominuje estymatę i ranking degeneruje się do rankingu `Q(s_H)` — w symulacji
  `tail=lin` schodzi na Swimmerze poniżej zera. Kalibracja: dopasowanie afiniczne
  `pozostały_zwrot ≈ a · (T − H)(1 − γ) · Q(s_H) + b` na osobnikach ocenionych w
  całości, czyli na ε-podzbiorze, który gate i tak opłaca. Wtedy
  `σ(H) = |a| · (T − H)(1 − γ) · std_i Q_i(s_H, ·)`.
- `σ(H)` maleje z `(T − H)` z konstrukcji, więc długi rollout wygasza nawet zły
  krytyk (Swimmer).
- Wszystkie kroki trafiają do replay buffera; to realne przejścia.

## 2. Dobór H: alokacja budżetu według ryzyka błędnej selekcji

GA nie potrzebuje dokładnego fitnessu, tylko poprawnego rankingu przy progu elit
i w turniejach. Kroki env powinny iść tam, gdzie najbardziej zmniejszają ryzyko
złej selekcji, nie tam, gdzie σ jest po prostu największa.

```
B      = budżet kroków na generację (np. ρ·N·T, równy dzisiejszemu budżetowi realnych ewaluacji)
Δ      = porcja rolloutu (25–50 kroków),  H_min = jedna porcja dla każdego
F_cut  = fitness na pozycji elite_count w bieżącym rankingu F̂

faza 0: każdy osobnik dostaje H_min  →  F̂, σ dla wszystkich
pętla, dopóki budżet:
    p_j = Φ( −|F̂_j − F_cut| / σ_j )       # P(j jest po złej stronie progu elit)
    j*  = argmax_j p_j   (spośród niezakończonych, H_j < T)
    dołóż Δ kroków osobnikowi j*, przelicz F̂_j*, σ_j*, F_cut
```

- Osobnik daleko pod progiem z dużą σ nie dostaje kroków (`p_j ≈ 0`). Osobnik
  tuż przy progu dostaje, aż σ zmaleje albo `H = T`. To racing / successive
  halving z krytykiem jako priorem.
- Baseline do ablacji jest wbudowany: ten sam `B` rozdzielony po równo
  (`H = B/N`) albo losowo (dzisiejszy gate: część osobników w całości, reszta
  `H_min`). Adaptacyjny vs równy przy identycznym budżecie to wprost test
  „niepewność czy budżet”.
- ε-frakcja osobników nadal dostaje pełny rollout losowo (nieobciążony pomiar
  błędu surogatu, uczenie β).
- Wariant bez sekwencyjności (na start): faza 0 dla wszystkich, jedna alokacja
  reszty budżetu proporcjonalnie do `p_j`, drugi przebieg. Wersja sekwencyjna
  wymaga N kopii env albo zapisu stanu fizyki (MuJoCo i DMC to umożliwiają).

## 3. Co zostaje z dzisiejszego SC-ERL

Ensemble, LCB, adaptacyjna β, ε-próbkowanie, normalizacja EMA. Zmienia się
`_gated_evaluation`: zamiast maski real/surogat zwraca wektor H i liczy fitness
ze wzoru wyżej. `rollout_policy` musi umieć zatrzymać się po H krokach i oddać
`s_H`, sumę nagród i `done`.

## 4. Umocowanie i ryzyka

- STEVE (Buckman 2018) / MVE (Feinberg 2018) przeniesione z aktualizacji krytyka
  na ewaluację fitnessu; po stronie EA racing / successive halving z priorem.
  Novelty w połączeniu i w motywacji z dekompozycji błędu.
- Środowiska, w których o rankingu decyduje późna faza epizodu, wymuszą duże H.
- Minimalne `H > 0` oznacza, że każdy osobnik coś kosztuje: przy `N=10`,
  `H_min=100` to ~1 realna ewaluacja na generację, tyle co dziś `ρ=0.1`.

## 5. Symulacja offline

`scripts/simulate_h_bootstrap.py` na checkpointach z 2026-09-07 (Swimmer-v5,
Hopper-v5, seed 0, populacja 100, 1 epizod na osobnika, β = 0). Pełne rollouty
raz, potem dowolne H liczone offline. Strategie przy tym samym budżecie `B`
(ułamek kosztu pełnej ewaluacji całej populacji):

- `uniform`: `H_j = B/N`.
- `random_gate`: `H_min` dla wszystkich, reszta budżetu na losowo wybrane pełne
  rollouty (dzisiejszy gate).
- `sigma_greedy`: porcje do `argmax σ_j` (bez progu elit).
- `adaptive`: porcje do `argmax p_j` (sekcja 2).

Metryki: Spearman(F̂, zwrot), precyzja top-10 (elity), regret selekcji
(średni realny zwrot prawdziwej top-10 minus wybranej top-10; mniej = lepiej).

### 5a. Sam estymator: stałe H dla wszystkich

Spearman(F̂, realny zwrot); `koszt` to ułamek kosztu pełnej ewaluacji populacji.

| | H=0 | H=25 | H=50 | H=100 | H=200 | H=500 | pełna |
|---|---|---|---|---|---|---|---|
| **Swimmer 66k** koszt | 0.00 | 0.03 | 0.05 | 0.10 | 0.20 | 0.50 | 1.00 |
| `tail=cal` | 0.045 | 0.383 | 0.280 | 0.077 | 0.576 | **0.878** | 1.00 |
| `tail=disc` | 0.045 | 0.177 | 0.294 | 0.421 | 0.491 | 0.539 | 0.539 |
| `tail=lin` | 0.045 | −0.196 | −0.188 | −0.264 | −0.220 | −0.134 | 1.00 |
| **Swimmer 110k** | | | | | | | |
| `tail=cal` | 0.086 | 0.248 | 0.589 | 0.583 | 0.729 | **0.908** | 1.00 |
| `tail=disc` | 0.086 | 0.249 | 0.408 | 0.641 | 0.710 | 0.725 | 0.725 |
| `tail=lin` | 0.086 | −0.127 | −0.449 | −0.397 | −0.471 | −0.251 | 1.00 |

Na Hopperze epizody trwają 70–83 kroki, więc `H = 100` to już pełna ewaluacja i
taniego reżimu nie ma. W przedziale `H = 5…50` (koszt 0.08–0.69) Spearman rośnie
z 0.11 do 0.70–0.80 na checkpoincie 60k, a na 100k dopiero `H = 50` wychodzi na
plus (0.29–0.33).

Wniosek: **sam H-bootstrap to duży zysk względem dzisiejszego surogatu**. Swimmer
110k: 0.086 → 0.908 przy połowie kosztu pełnej ewaluacji, regret selekcji 23.5 → 1.8.
Zysk jest tym większy, im dłuższe epizody względem tempa, w jakim ranking się
klaruje. Epizody DMC dog są 1000-krokowe i nieterminujące, tak jak Swimmer.

### 5b. Alokacja budżetu (estymator `tail=cal`, koszt ε-podzbioru wliczony)

Spearman / regret selekcji przy równym budżecie.

| budżet | strategia | Hopper 60k | Hopper 100k | Swimmer 66k | Swimmer 110k |
|---|---|---|---|---|---|
| 0.25 | uniform | 0.404 / 68.9 | 0.184 / 32.8 | 0.573 / 31.2 | **0.781** / 9.8 |
| 0.25 | random_gate (dziś) | 0.403 / 10.4 | 0.184 / 32.8 | 0.409 / 22.5 | 0.437 / 25.4 |
| 0.25 | sigma_greedy | 0.421 / **6.8** | 0.139 / 32.8 | **0.656** / 27.0 | 0.613 / 11.9 |
| 0.25 | adaptive | **0.490** / 44.0 | 0.137 / 32.8 | 0.464 / 26.0 | 0.524 / **7.5** |
| 0.50 | uniform | 0.532 / 38.3 | 0.036 / 34.9 | **0.905** / 9.2 | **0.932** / **3.2** |
| 0.50 | random_gate (dziś) | 0.563 / 11.5 | 0.223 / 20.6 | 0.583 / 16.0 | 0.584 / 8.2 |
| 0.50 | sigma_greedy | 0.647 / 14.5 | 0.243 / 27.0 | 0.769 / 11.6 | 0.867 / 5.7 |
| 0.50 | adaptive | **0.668** / **9.8** | **0.245** / **18.5** | 0.660 / **7.0** | 0.724 / 3.9 |

- **Dzisiejszy binarny gate (`random_gate`) nigdy nie wygrywa na jakości rankingu.**
  Przy tym samym budżecie ciągłe H bije go wszędzie, na Swimmerze wyraźnie
  (0.93 vs 0.58 przy budżecie 0.5). To najmocniejszy wynik dla artykułu: problemem
  jest binarność decyzji, nie estymator niepewności.
- **Hopper (epizody terminujące, zmiennej długości): alokacja sterowana
  niepewnością wygrywa.** `adaptive` bije `uniform` na obu metrykach i obu
  checkpointach (0.668 vs 0.532 i regret 9.8 vs 38.3 przy budżecie 0.5).
- **Swimmer (epizody stałej długości): równe H jest już bliskie optimum** dla
  rankingu globalnego, a `adaptive` daje tylko niższy regret selekcji przy małym
  budżecie (7.5 vs 9.8). To spójne z mechanizmem: przy stałej długości epizodu
  każdy osobnik ma tę samą strukturę niepewności, więc nie ma czego różnicować.

### 5c. Zastrzeżenia

- Jeden seed, dwa środowiska, jeden epizod na osobnika, populacje wyłącznie z
  runów `surrogate.mode=ensemble` (100k kroków, lokalne). To symulacja offline
  jakości rankingu w pojedynczej generacji, nie wynik treningowy — nie mierzy
  sprzężenia zwrotnego między selekcją a uczeniem krytyka.
- `real` liczone z jednego deterministycznego resetu (`seed=0`), więc jest
  1-próbkowym estymatorem prawdziwego fitnessu. Tak samo działa dziś GA
  (`eval_trials=1` na Swimmerze), więc porównanie jest uczciwe, ale bezwzględne
  wartości regretu są obarczone tym szumem.
- Kalibracja korzysta z 10% osobników ocenionych w całości; ich koszt jest
  wliczony w budżet. Przy `budget = 0.10` na Swimmerze sam ε-podzbiór zjada cały
  budżet, dlatego tych wierszy nie ma w tabeli.
- Reprodukcja: `uv run --extra mujoco-envs python scripts/simulate_h_bootstrap.py
  checkpoints/*.pt`. Trajektorie są cache'owane w `outputs/h_bootstrap_cache/`.

### 5d. Co z tego wynika dla dalszych prac

1. Wymienić binarny gate na ciągłe H — to broni się niezależnie od tego, czy
   niepewność steruje alokacją.
2. Ogon musi być kalibrowany na ε-podzbiorze; bez tego mechanizm szkodzi.
3. Alokacja sterowana niepewnością ma sens tam, gdzie epizody terminują w różnych
   momentach (Hopper, Walker2d, Ant). Przy stałej długości (Swimmer, DMC dog)
   równe H wystarcza — i to jest testowalna predykcja, a nie wymówka.
4. Następny krok przed pisaniem: powtórzyć 5b na 3–5 seedach i dołożyć
   `dm_control/dog-walk-v0`, bo to środowisko, na którym stoi teza artykułu.

## 6. Implementacja online (2026-09-08)

`surrogate.gate_mode=h_bootstrap` (domyślny). Binarny gate (`topk`, `relative`)
zostaje jako baseline — porównanie przy równym budżecie jest tezą artykułu.
Kod: `src/common/h_bootstrap.py` (`EpisodeRunner`, `TailCalibrator`,
`stop_probability`) + `SurrogateController._h_bootstrap_evaluation`.

| decyzja | wybór |
|---|---|
| alokacja H | stopping online, jeden przebieg, jeden env, bez snapshotów |
| tryby surogatu | wszystkie 4; `random` → σ=0 → alokacja `uniform` |
| budżet generacji | `B = ρ · N · L_ema`, ρ i jego adaptacja bez zmian |
| ogon | kalibrowany (`tail=cal`), fallback `a=b=0` → sam zwrot cząstkowy |
| ε-osobnicy | pełny rollout niezależnie od reguły stopu |

### Czym różni się od symulacji

- **Brak przeplatania.** Symulacja dokłada porcję dowolnemu osobnikowi w
  dowolnym momencie, bo ma trajektorie w pamięci. Jedna instancja `gym.Env`
  nie umie wznowić epizodu, więc każdy osobnik jest toczony raz, a niepewność
  decyduje **kiedy przerwać**: po każdej porcji `Δ` liczone jest
  `p_j = Φ(−|F̂_j − F_cut| / σ_j)` i przerwanie następuje przy `p_j < p_stop`.
- **`F_cut` jest częściowo z priora.** Startuje progiem elit z poprzedniej
  generacji; kwantyl z bieżącej generacji przejmuje dopiero, gdy oceniona jest
  co najmniej połowa populacji. Kolejność jest rosnąca po `p_j` liczonym z
  priora (σ na stanach z bufora), więc osobnicy przy progu trafiają na koniec,
  gdy `F_cut` jest już dobrze znany, i to im zostaje budżet.
- **`Δ` adaptuje się do budżetu**: `Δ = clip((B − n_ε·L_ema) / N, 1, h_chunk)`.
  Symulacja ma stałe `Δ = 25`, co na Hopperze (epizody 70–83 kroków) nie
  mieści się w budżecie równym dzisiejszemu gate'owi.
- **Budżet jest górnym ograniczeniem, nie celem.** Limit osobnika to równa
  część tego, co zostało: `H_cap = max(Δ, B_left / (ilu_zostało + 1))`. Kroki
  oddane przez tych, którzy stanęli wcześnie, kumulują się dla osobników przy
  progu na końcu kolejności, a niewykorzystana reszta budżetu po prostu nie
  jest wydawana. Zachłanne `B_left − Δ·ilu_zostało` oddawało cały budżet
  pierwszemu w kolejności — czyli, przy tej kolejności, najmniej potrzebującemu.
- **Horyzont `T`** z `env.spec.max_episode_steps`; DMC przez shimmy zwraca
  `None` (limit siedzi w dm_control), więc wtedy `T` rośnie do najdłuższego
  zaobserwowanego epizodu. `T` ≠ `L_ema`: `T` wchodzi do ogona, `L_ema` do
  budżetu, a różnicę (Hopper: 1000 vs ~80) wchłania współczynnik `a`.

### Zimny start

Dopóki `TailCalibrator` nie ma `min_pairs = 30` par, `a = b = 0` — fitness to
sam zwrot cząstkowy, `σ = 0`. Żeby to nie trwało, w tym stanie połowa budżetu
idzie na pełne rollouty (`n_ε = 0.5·B / L_ema`), z których harvestowane są pary
co `Δ` kroków wzdłuż trajektorii. Przy `warmup_steps` ustawionym wysoko
(Hopper 25k, DMC dog 100k) pierwsze generacje i tak są pełne i kalibrują ogon
od razu; przy `sc_erl.yaml` (`warmup_steps: 10`, czyli Swimmer/HalfCheetah/Ant)
kalibracja rozkręca się z samego budżetu w 1–2 generacjach.

### Metryki

`h_mean`, `h_median`, `h_full_frac`, `sigma_tail_mean`, `tail_a`, `tail_b`,
`cal_pairs`, `budget_steps`, `episode_len_ema`, `f_cut`, `beta`. Zachowane:
`rho`, `e_hat_mean`, `gate_auc_u`, `gate_spearman` — te dwie ostatnie mierzą
teraz to, czym są: czy σ ogona przewiduje błąd estymaty `|F̂(h) − R|`.

### Weryfikacja estymatora

```bash
uv run --extra mujoco-envs python scripts/simulate_h_bootstrap.py \
    --verify checkpoints/ckpt_Hopper-v5_seed0_100000.pt
```

Sprawdza na checkpointach, że produkcyjny `TailCalibrator` daje tę samą
formułę afiniczną i to samo `σ` co referencja offline, że przy `H = długość
epizodu` wychodzi dokładnie realny zwrot i `σ = 0`, oraz drukuje Spearmana
online obok offline dla siatki `H`.
