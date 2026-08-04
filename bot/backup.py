"""Daily backup — tgz rotation of data/, logs/, config/, models/."""

import logging
import tarfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKUP_DIR = BASE_DIR / "backups"
SOURCES = ["data", "logs", "config", "models"]
MAX_BACKUPS = 7

# Never archive plaintext secrets. Backups are unencrypted tar.gz files kept on
# disk for 7 days, so including credentials.ini would leak the live MT5 login
# (agent audit H3). Match on filename so nested copies are excluded too.
EXCLUDE_NAMES = {"credentials.ini"}


def _exclude_secrets(tarinfo):
    if Path(tarinfo.name).name in EXCLUDE_NAMES:
        logging.info(f"Backup: excluding secret file {tarinfo.name}")
        return None
    return tarinfo


def run_backup():
    BACKUP_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"doto_backup_{date_str}.tar.gz"
    archive_path = BACKUP_DIR / archive_name
    with tarfile.open(archive_path, "w:gz") as tar:
        for src in SOURCES:
            src_path = BASE_DIR / src
            if src_path.exists():
                tar.add(src_path, arcname=src, filter=_exclude_secrets)
    logging.info(f"Backup created: {archive_path} ({archive_path.stat().st_size / 1024:.1f} KB)")
    _rotate()


def _rotate():
    backups = sorted(BACKUP_DIR.glob("doto_backup_*.tar.gz"), reverse=True)
    for old in backups[MAX_BACKUPS:]:
        old.unlink()
        logging.info(f"Removed old backup: {old.name}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_backup()


if __name__ == "__main__":
    main()
