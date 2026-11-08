# Third-party notices

The evaluation harness is MIT licensed; see LICENSE. This does not relicense
third-party datasets, models, runtimes, simulator assets, or external sources.

- **DexJoCo**: https://github.com/brave-eai/dexjoco at commit
  `8d23b0fab23b17a58c4b55f3942e17013aaf8267`. The source uses MIT;
  bundled upstream components/assets may have separate terms. Obtain the
  simulator directly upstream and retain its license/notices.
- **DexJoCo-Datasets-Raw**: https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-Raw
  at revision `125df5c1019e97503929ef1a0ad8f90373436afa`. The pinned dataset
  card declares MIT. Frozen knowledge contains selected derived training
  images and numeric motion/reference data. A copy of the dataset card is
  included under third_party/ as attribution; per-task SOURCES.md records
  demonstration and external-source provenance.
- **Codex CLI and model access** are external prerequisites, not bundled or
  relicensed by this release. Availability and service terms are separate.
- Python and OS dependencies retain their own licenses. They are installed
  separately and are not vendored.

Robot-reference provenance includes Franka Emika Panda (Apache-2.0), Wonik
Allegro (BSD-2-Clause), and Allegro URDF material obtained through the upstream
GeoRT assets. GeoRT's supplied license is CC BY-NC and restricts commercial
use. Copies of these notices are in third_party/. Robot meshes/URDF files are
obtained from the external upstream checkout at runtime, not bundled here;
some frozen helpers contain derived kinematic values. The harness MIT license
does not override these source terms or grant commercial rights to GeoRT
material. Keep these notices and the per-task source records when redistributing
the frozen knowledge.
