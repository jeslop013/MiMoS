# Emission inputs: sources

## hot_ef.csv: hot emission factors

- **Source**: EMEP/EEA air pollutant emission inventory guidebook, *1.A.3.b.i-iv Road Transport Appendix 4 Emission Factors*, sheet `HOT_EMISSIONS_PARAMETERS`, guidebook edition v2026 (linked to COPERT 5.10.1).
  - Page: https://www.eea.europa.eu/en/analysis/publications/emep-eea-guidebook-2023/part-b-sectoral-guidance-chapters/1-energy/1-a-combustion/1-a-3-b-i-1
- **Kept**: the rows for passenger cars, light commercial vehicles and L-category (mopeds and motorcycles), on flat roads (slope 0 or empty), with empty load (load applies to heavy vehicles only). Coefficients are copied unchanged. Cells written as text (`'= 0.03'`) or as references to another row of the same column were resolved to their values.
- **Licence**: EEA content may be reused with acknowledgement (EEA standard re-use policy).

## fleet_example.csv: example fleet composition

A made-up fleet, an example of the optional `fleet.csv` input (docs/input_format.md). It is used when a scenario has no fleet of its own, for example the example town. Like the town's behavioural constants, its values are round and plausible, not measured.

- **Classes**: passenger cars 80%, light commercial vehicles (vans) 10%, motorcycles 7%, mopeds 3%; one representative EEA segment per class (cars Medium, vans N1-II, motorcycles 4-stroke 250-750 cm³, mopeds < 50 cm³).
- **Fuels**: cars petrol 47%, diesel 45%, battery electric 5%, LPG 3%; vans diesel 85%, petrol 13%, battery electric 2%.
- **Emission standards** (cars and vans): pre-Euro 2%, Euro 1 3%, Euro 2 7%, Euro 3 13%, Euro 4 18%, Euro 5 17%, Euro 6 a/b/c 20%, Euro 6 d-temp 10%, Euro 6 d/e 10%. Pre-Euro petrol cars use the ECE 15/04 factors.
- `share` = class share x fuel share x standard share; MiMoS normalises the shares.
