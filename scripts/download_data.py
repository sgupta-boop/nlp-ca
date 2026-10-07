"""Phase 0: download every dataset into data/raw/.

Run from the project root:
    python scripts/download_data.py            # everything
    python scripts/download_data.py abt_buy    # one dataset

Each step is skipped if its output already exists, so the script is safe to re-run.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Open Food Facts rejects the default python-requests agent (HTTP 403) and asks for a named one
USER_AGENT = "product-nlp-lab-project/0.1 (student coursework)"
WDC_BASE ="https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/"


def fetch(url: str, dest: Path, max_retries: int = 30) -> Path:
    """Stream a URL to disk (skips if the file already exists)."""
    if dest.exists():
        print(f"  exists: {dest.name}")
        return dest
    print(f"  downloading {url}")
    # Write to .part and rename only when complete, so a failed download never looks finished.
    # If the connection drops, retry and continue from the bytes already saved (HTTP Range).
    part = dest.with_name(dest.name + ".part")
    for attempt in range(max_retries):
        done = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:  # nothing left to fetch
                    break
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"  # 200 = server ignored Range
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            print(f"  connection dropped at {part.stat().st_size >> 20} MB, retry {attempt + 1}: {e.__class__.__name__}")
    else:
        raise RuntimeError(f"gave up after {max_retries} retries")
    part.replace(dest)
    return dest


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def kaggle_dataset(slug: str, out_dir: Path) -> None:
    """Kaggle REST API with the access token in ~/.kaggle/access_token
    (Kaggle -> Settings -> API -> Generate New Token)."""
    if out_dir.exists() and any(out_dir.glob("*.csv")):
        print(f"  exists: {out_dir.name}")
        return
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    z = RAW / f"{out_dir.name}.zip"
    print(f"  downloading {url}")
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(z, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    unzip(z, out_dir)


def bigbasket():
    kaggle_dataset("surajjha101/bigbasket-entire-product-list-28k-datapoints", RAW / "bigbasket")


def flipkart():
    kaggle_dataset("PromptCloudHQ/flipkart-products", RAW / "flipkart")


def abt_buy():
    z = fetch("https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip", RAW / "Abt-Buy.zip")
    unzip(z, RAW / "abt_buy")


def amazon_google():
    z = fetch("https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip",
              RAW / "Amazon-GoogleProducts.zip")
    unzip(z, RAW / "amazon_google")


def wdc_products():
    for name in ["80pair", "80multi", "50pair", "50multi", "20pair", "20multi",
                 "val_pair", "val_multi"]:
        z = fetch(WDC_BASE + f"{name}.zip", RAW / f"wdc_{name}.zip")
        unzip(z, RAW / "wdc_products")


def wdc_pave():
    repo = RAW / "wdc_pave_repo"
    if repo.exists():
        print("  exists: wdc_pave_repo")
        return
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/wbsg-uni-mannheim/wdc-pave.git", str(repo)], check=True)


def aksharantar_hindi():
    z = fetch("https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/hin.zip",
              RAW / "aksharantar_hin.zip")
    unzip(z, RAW / "aksharantar_hin")


def google_taxonomy():
    fetch("https://www.google.com/basepages/producttype/taxonomy.en-US.txt",
          RAW / "google_taxonomy.en-US.txt")


def open_food_facts_india():
    """Download the full Open Food Facts CSV export (~1.3 GB gzip, tab-separated), keep only
    rows sold in India, and delete the big file. DuckDB streams the file, so it never has to fit
    in RAM. (Querying the Hugging Face Parquet copy remotely was rate-limited with HTTP 429.)"""
    out = RAW / "off_india.parquet"
    if out.exists():
        print("  exists: off_india.parquet")
        return
    gz = fetch("https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
               RAW / "off_products.csv.gz")
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    tmp = out.with_name(out.name + ".part")
    con.execute(f"""
        COPY (
            SELECT code, product_name, brands, categories, quantity, countries_tags
            FROM read_csv('{gz.as_posix()}', delim='\t', quote='', header=true,
                          all_varchar=true, ignore_errors=true, max_line_size=10000000)
            WHERE countries_tags LIKE '%en:india%'
        ) TO '{tmp.as_posix()}' (FORMAT PARQUET)
    """)
    tmp.replace(out)
    n_rows = con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    print(f"  saved {out.name}: {n_rows} rows")
    gz.unlink()  # 1.3 GB; re-downloadable


STEPS = {
    "bigbasket": bigbasket, "flipkart": flipkart, "abt_buy": abt_buy,
    "amazon_google": amazon_google, "wdc_products": wdc_products, "wdc_pave": wdc_pave,
    "aksharantar_hindi": aksharantar_hindi, "google_taxonomy": google_taxonomy,
    "open_food_facts_india": open_food_facts_india,
}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(STEPS)
    failed = []
    for name in chosen:
        print(f"[{name}]")
        try:
            STEPS[name]()
        except Exception as e:  # keep going so one failure does not block the rest
            print(f"  FAILED: {e}")
            failed.append(name)
    print("failed:", failed or "none")
