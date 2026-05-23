#!/usr/bin/env bash
# Download the four public CAN-IDS datasets used by mids-bench Batch 3.
#
# Two are direct (wget-friendly) and two require a quick browser step:
#
#   ROAD     -> direct (Zenodo, ~557 MB)
#   OTIDS    -> direct (Dropbox folder, ~150 MB)
#   CT&T     -> direct (DTU bulk archive, ~1 GB)
#   CrySyS   -> browser (figshare collection has multiple zip articles;
#               point-and-click is faster than scripting the figshare API)
#
# Edit ROOT below to where you want everything. Defaults to /root/autodl-tmp,
# matching the YAML configs in configs/data/.

set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
mkdir -p "$ROOT"

echo "Downloading into $ROOT"
echo "================================================================"

# ---------------------------------------------------------------------
# 1) ROAD (Zenodo, 556.7 MB, single road.zip)
# ---------------------------------------------------------------------
echo
echo "[1/4] ROAD  ->  $ROOT/road/"
mkdir -p "$ROOT/road"
if [ ! -f "$ROOT/road/road.zip" ]; then
    wget -c -O "$ROOT/road/road.zip" \
        "https://zenodo.org/records/10462796/files/road.zip?download=1"
fi
echo "  unzipping..."
unzip -q -o "$ROOT/road/road.zip" -d "$ROOT/road/"
echo "  ROAD ready."

# ---------------------------------------------------------------------
# 2) OTIDS (HCRL, hosted on Dropbox folder share, ~150 MB)
# ---------------------------------------------------------------------
# The Dropbox UI gives a folder share with dl=0 (preview); flipping to
# dl=1 returns a single zip of the folder.
echo
echo "[2/4] OTIDS  ->  $ROOT/otids/"
mkdir -p "$ROOT/otids"
if [ ! -f "$ROOT/otids/otids.zip" ]; then
    wget -c -O "$ROOT/otids/otids.zip" \
      "https://www.dropbox.com/scl/fo/8kll7yvbgogkp0vahowvm/ADhDIC8LRFL8wHUexib3C3w?rlkey=8cp7scxgw25yt4wp8v2c2v8mp&dl=1"
fi
echo "  unzipping..."
unzip -q -o "$ROOT/otids/otids.zip" -d "$ROOT/otids/"
echo "  OTIDS ready (expect 4 .txt files: Attack_free, DoS, Fuzzy, Impersonation)."

# ---------------------------------------------------------------------
# 3) CT&T  (DTU Data, article 24805533, version 1, ~1 GB total)
# ---------------------------------------------------------------------
# DTU Data exposes a bulk archive URL for any published article via
# /ndownloader/articles/<id>/versions/<v>; that returns a zip of all files.
echo
echo "[3/4] CT&T  ->  $ROOT/ctnt/"
mkdir -p "$ROOT/ctnt"
if [ ! -f "$ROOT/ctnt/ctnt.zip" ]; then
    wget -c -O "$ROOT/ctnt/ctnt.zip" \
      "https://data.dtu.dk/ndownloader/articles/24805533/versions/1"
fi
echo "  unzipping..."
unzip -q -o "$ROOT/ctnt/ctnt.zip" -d "$ROOT/ctnt/"
echo "  CT&T ready (per-vehicle subfolders: chevy_silverado, chevy_traverse, ...)."

# ---------------------------------------------------------------------
# 4) CrySyS — manual step (figshare COLLECTION; multiple article zips)
# ---------------------------------------------------------------------
echo
echo "[4/4] CrySyS  ->  $ROOT/crysys/   (manual step)"
mkdir -p "$ROOT/crysys"
cat <<EOF
   The CrySyS dataset is a figshare COLLECTION (DOI 10.6084/m9.figshare.c.6726165),
   which is a wrapper around several individual articles, each shipping its own
   zip. There is no single bulk-download URL.

   Open this in your browser:
     https://springernature.figshare.com/collections/CrySyS_dataset_of_CAN_traffic_logs_containing_fabrication_and_masquerade_attacks/6726165

   For each article in the collection, click "Download all" -> save to:
     $ROOT/crysys/

   Then unzip everything in place:
     cd $ROOT/crysys && for f in *.zip; do unzip -q -o "\$f"; done

EOF

echo
echo "================================================================"
echo "When CrySyS is in place, sanity-check the four roots:"
echo "  ls $ROOT/road/ $ROOT/otids/ $ROOT/ctnt/ $ROOT/crysys/"
echo
echo "Then run a smoke test per dataset:"
echo "  python scripts/run_one.py --model mids --dataset otids  --fold 0"
echo "  python scripts/run_one.py --model mids --dataset road   --fold 0"
echo "  python scripts/run_one.py --model mids --dataset ctnt   --fold 0"
echo "  python scripts/run_one.py --model mids --dataset crysys --fold 0"
